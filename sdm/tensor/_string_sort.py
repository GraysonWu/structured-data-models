# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact stable string sorting with native PyTorch operations.

Seven UTF-8 bytes, each stored as ``byte + 1`` in nine bits, form one word;
zero marks the end of a string, so prefixes sort before their extensions.
An LSD radix sort orders strings by their first ``PREFIX`` bytes while
tracking prefix ties. Longer strings that tie on the prefix are ranked by
their remaining bytes with block-pair prefix doubling. The same static-shape
code runs eagerly and under ``torch.compile``.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from sdm.tensor.string import StringTensor

PREFIX = 21


def argsort(values: StringTensor, descending: bool) -> Tensor:
    """Return a stable sort permutation of one-dimensional strings.

    Nulls sort last in both directions.
    """
    n = values.numel()
    if values.stride(0) == 0:
        # Every row refers to the same string.
        return torch.arange(n, device=values.device)

    dtype = _index_dtype(max(values._offset.numel(), values._data.numel() + 1))
    rows = torch.arange(n, device=values.device, dtype=dtype)
    rows = rows * values.stride(0) + values._logical_storage_offset
    start = values._offset[rows].to(dtype)
    length = values._offset[rows + 1].to(dtype) - start
    valid = None if values._valid is None else values._valid[rows]
    if valid is not None:
        # Null payloads compare as empty and keep their input order.
        length = length.where(valid, 0)
    # Masked bytes read any valid position, so empty buffers get one byte.
    data = values._data if values._data.numel() else values._data.new_zeros(1)
    bound = values._max_length
    if bound is None:
        bound = values._data.numel()

    # LSD radix sort over the prefix, least significant word first.
    prefix = min(max(bound, 1), PREFIX)
    perm = rank = None
    for depth in reversed(range(0, prefix, 7)):
        rows = slice(None) if perm is None else perm
        key = _word(
            data, start[rows], length[rows], depth, min(7, prefix - depth)
        )
        if depth == 0 and valid is not None:
            key = key.where(
                valid[rows], -1 if descending else torch.iinfo(key.dtype).max
            )
        key, order = key.sort(descending=descending, stable=True)
        perm = order if perm is None else perm[order]
        if bound > prefix:
            # Rank the prefix bytes processed so far in sorted order.
            changed = torch.ones_like(key, dtype=torch.bool)
            changed[1:] = key[1:] != key[:-1]
            if rank is not None:
                rank = rank[order]
                changed[1:] |= rank[1:] != rank[:-1]
            rank = changed.cumsum(0)
    assert perm is not None  # The prefix spans at least one word.
    if rank is None:
        return perm

    # Only longer strings that tie on the prefix need their remaining bytes.
    tied = torch.zeros_like(rank, dtype=torch.bool)
    tied[1:] = rank[1:] == rank[:-1]
    tied[:-1] |= tied[1:].clone()
    tied &= length[perm] > prefix
    refine = functools.partial(_refine, bound=bound, descending=descending)
    if torch.compiler.is_compiling():
        # Static bounds: at most total // (PREFIX + 1) strings are longer,
        # and their tails span at most total // 7 seven-byte blocks.
        total = values._data.numel()
        slots = min(n, total // (PREFIX + 1))
        blocks = slots * max(1, -(-(bound - PREFIX) // 7))
        refine = functools.partial(
            refine, slots=slots, blocks=min(total // 7, blocks)
        )
        operands = (data, start, length, perm, rank, tied)
        return torch.cond(tied.any(), refine, _keep, operands)

    # Eager: groups of identical strings are already in order.
    tied &= _mixed(data, start, length, perm, rank, tied)
    if not tied.any():
        return perm
    blocks = ((length[perm] - PREFIX + 6) // 7)[tied]
    return refine(
        data,
        start,
        length,
        perm,
        rank,
        tied,
        slots=int(tied.sum()),
        blocks=int(blocks.sum()),
    )


def _keep(
    data: Tensor,
    start: Tensor,
    length: Tensor,
    perm: Tensor,
    rank: Tensor,
    tied: Tensor,
) -> Tensor:
    return perm.clone()


def _refine(
    data: Tensor,
    start: Tensor,
    length: Tensor,
    perm: Tensor,
    rank: Tensor,
    tied: Tensor,
    *,
    bound: int,
    descending: bool,
    slots: int,
    blocks: int,
) -> Tensor:
    """Order prefix ties by their remaining bytes.

    ``slots`` and ``blocks`` bound the tied strings and their tail blocks.
    """
    n = perm.numel()
    index = torch.nonzero_static(tied, size=slots, fill_value=n).squeeze(1)
    rows = torch.cat((perm, perm.new_zeros(1)))[index]
    used = index < n
    tail = _block_rank(
        data=data,
        start=start[rows] + PREFIX,
        length=(length[rows] - PREFIX).where(used, 0),
        bound=bound - PREFIX,
        capacity=blocks,
    )
    # Padding slots scatter to their own destinations past the rows.
    padding = n + torch.arange(slots, device=data.device)
    tail = tail.new_zeros(n + slots).scatter_(
        0, index.where(used, padding), tail
    )[:n]
    if descending:
        tail = slots - tail
    # Tail ranks are at most ``slots``, so the combined key stays ordered.
    key = rank * (slots + 1) + tail
    return perm[key.argsort(stable=True)]


def _mixed(
    data: Tensor,
    start: Tensor,
    length: Tensor,
    perm: Tensor,
    rank: Tensor,
    tied: Tensor,
) -> Tensor:
    """Whether each tied position's group holds different strings (eager)."""
    position = tied.nonzero().squeeze(1)
    group = rank[position]
    first = torch.ones_like(group, dtype=torch.bool)
    first[1:] = group[1:] != group[:-1]
    index = torch.arange(first.numel(), device=first.device)
    leader = perm[position[(index * first).cummax(0).values]]
    row = perm[position]
    count = length[row]
    differs = count != length[leader]
    differs |= _rows_differ(
        data=data,
        start=start[row] + PREFIX,
        other=start[leader] + PREFIX,
        count=(count - PREFIX).where(~differs, 0),
    )
    # Any difference within a group, via prefix sums over the groups.
    bounds = torch.cat((first, first.new_ones(1))).nonzero().squeeze(1)
    seen = torch.cat((group.new_zeros(1), differs.cumsum(0)))
    mixed = (seen[bounds].diff() > 0)[first.cumsum(0) - 1]
    return torch.zeros_like(tied).index_put_((position,), mixed)


def _rows_differ(
    data: Tensor,
    start: Tensor,
    other: Tensor,
    count: Tensor,
) -> Tensor:
    """Whether ``count`` bytes at ``start`` differ from those at ``other``.

    Bytes are compared in four chunks so temporaries stay near the data size.
    """
    dtype = start.dtype
    first = torch.cat((count.new_zeros(1), count.cumsum(0, dtype=dtype)))
    total = int(first[-1])
    differs = torch.zeros_like(count, dtype=torch.bool)
    step = max(1, -(-total // 4))
    for begin in range(0, total, step):
        byte = torch.arange(
            begin, min(begin + step, total), device=data.device, dtype=dtype
        )
        row = torch.searchsorted(
            first, byte, right=True, out_int32=dtype == torch.int32
        ).sub_(1)
        byte -= first[row]
        mismatch = data[start[row] + byte] != data[other[row] + byte]
        differs[row[mismatch]] = True
    return differs


def _word(
    data: Tensor,
    start: Tensor,
    length: Tensor,
    depth: int,
    width: int = 7,
) -> Tensor:
    # Up to three nine-bit symbols fit in int32; more need int64.
    dtype = torch.int32 if width <= 3 else torch.int64
    byte = torch.arange(
        depth, depth + width, device=data.device, dtype=start.dtype
    )
    absent = byte >= length[:, None]
    index = start[:, None] + byte  # [rows, width]
    index.masked_fill_(absent, data.numel() - 1)
    symbol = data[index]
    del index
    symbol = symbol.to(dtype).add_(1).masked_fill_(absent, 0)
    symbol <<= torch.arange(9 * (width - 1), -1, -9, device=data.device)
    return symbol.sum(-1, dtype=dtype)


def _index_dtype(size: int) -> torch.dtype:
    # Positions use int32 unless the indexed extent needs int64.
    return torch.int32 if size < torch.iinfo(torch.int32).max else torch.int64


def _layout(count: Tensor, capacity: int) -> tuple[Tensor, Tensor, Tensor]:
    """Map ``capacity`` slots to rows holding ``count`` slots each.

    Unused trailing slots belong to a padding row with index ``count.numel()``.
    """
    dtype = _index_dtype(capacity + 1)
    count = count.to(dtype)
    first = torch.cat((count.new_zeros(1), count.cumsum(0, dtype=dtype)))
    rows = torch.arange(count.numel() + 1, device=count.device, dtype=dtype)
    repeats = torch.cat((count, (capacity - first[-1]).reshape(1)))
    owner = rows.repeat_interleave(repeats, output_size=capacity)
    local = torch.arange(capacity, device=count.device, dtype=dtype)
    return first, owner, local - first[owner]


def _dense_rank(key: Tensor) -> Tensor:
    # Rank zero is reserved for key zero, which marks an empty block.
    key = torch.cat((key, key.new_zeros(1)))
    sorted_key, order = key.sort()
    dtype = _index_dtype(key.numel())
    step = (sorted_key.diff() != 0).to(dtype).cumsum(0, dtype=dtype)
    step = torch.cat((step.new_zeros(1), step))
    return torch.empty_like(step).scatter_(0, order, step)[:-1]


def _block_rank(
    data: Tensor,
    start: Tensor,
    length: Tensor,
    bound: int,
    capacity: int,
) -> Tensor:
    """Rank rows by merging adjacent seven-byte blocks with static shapes.

    ``capacity`` must cover the total number of blocks.
    """
    n = start.numel()
    # Rows with an empty tail (padding) own no blocks.
    count = (length + 6) // 7
    first, owner, local = _layout(count, capacity)
    real = owner < n
    start = torch.cat((start, start.new_zeros(1)))
    length = torch.cat((length, length.new_zeros(1)))
    byte = start[owner] + 7 * local.to(start.dtype)
    end = (start + length)[owner]
    word = torch.zeros(capacity, dtype=torch.int64, device=data.device)
    for i in range(7):
        present = byte + i < end
        symbol = data[(byte + i).where(present, data.numel() - 1)].long()
        word |= (symbol + 1).where(present, 0) << (54 - 9 * i)
    rank = _dense_rank(word)

    # [blocks] -> [ceil(blocks / 2)] per row until one block remains.
    for _ in range(math.ceil(math.log2(max(1, -(-bound // 7))))):
        merged = (count + 1) // 2
        merged_capacity = capacity // 2 + n
        merged_first, owner, local = _layout(merged, merged_capacity)
        real = owner < n
        first = torch.cat((first, first[-1:]))
        count = torch.cat((count, count.new_zeros(1)))
        left = (first[owner] + 2 * local).where(real, capacity)
        has_right = real & (2 * local + 1 < count[owner])
        right = (left + 1).where(has_right, capacity)
        rank = torch.cat((rank, rank.new_zeros(1)))
        rank = _dense_rank(rank[left].long() * (capacity + 1) + rank[right])
        count, capacity, first = merged, merged_capacity, merged_first
    return rank[first[:-1]]
