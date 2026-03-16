"""Data loading and tokenization utilities for LoopLM pre-training."""

import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, TensorDataset


def make_tokenizer(model_id: str = "HuggingFaceTB/SmolLM2-135M"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def tokenize_and_chunk(texts: list[str], tokenizer, seq_len: int) -> Tensor:
    """Tokenize a list of strings and split into fixed-length chunks.

    Returns:
        (N, seq_len + 1) LongTensor — each row is a training example.
        The trainer will use [:, :-1] as input and [:, 1:] as targets.
    """
    all_ids: list[int] = []
    eos = tokenizer.eos_token_id or 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            all_ids.extend(ids)
            all_ids.append(eos)  # document separator

    tokens = torch.tensor(all_ids, dtype=torch.long)
    chunk_len = seq_len + 1
    n_chunks = len(tokens) // chunk_len
    return tokens[: n_chunks * chunk_len].view(n_chunks, chunk_len)


def make_dataloader(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-v1",
    split: str = "train",
    tokenizer_id: str = "HuggingFaceTB/SmolLM2-135M",
    seq_len: int = 512,
    batch_size: int = 4,
    max_chunks: int | None = None,
    shuffle: bool = True,
    text_column: str = "text",
) -> DataLoader:
    """Load a HuggingFace text dataset and return a ready-to-use DataLoader.

    Tokenization and chunking are applied via datasets.map so results are
    cached to disk automatically — subsequent runs with the same args skip
    re-tokenization.

    Args:
        max_chunks: cap the number of training chunks (useful for quick runs)
    """
    from datasets import load_dataset

    tokenizer = make_tokenizer(tokenizer_id)
    eos = tokenizer.eos_token_id or 0
    chunk_len = seq_len + 1

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Step 1: tokenize each document → list of token IDs with EOS separator.
    # datasets caches this on disk keyed by the map function source + args.
    def _tokenize(batch):
        ids_per_doc = []
        for text in batch[text_column]:
            if text.strip():
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids_per_doc.append(ids + [eos])
            else:
                ids_per_doc.append([])
        return {"input_ids": ids_per_doc}

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Step 2: concatenate all token IDs across the batch and split into
    # fixed-length chunks of (seq_len + 1).
    def _chunk(batch):
        all_ids: list[int] = []
        for ids in batch["input_ids"]:
            all_ids.extend(ids)
        n = len(all_ids) // chunk_len
        chunks = [all_ids[i * chunk_len : (i + 1) * chunk_len] for i in range(n)]
        return {"input_ids": chunks}

    chunked = tokenized.map(
        _chunk,
        batched=True,
        desc="Chunking",
    )

    if max_chunks is not None:
        chunked = chunked.select(range(min(max_chunks, len(chunked))))

    chunked.set_format(type="torch", columns=["input_ids"])

    print(
        f"Dataset: {len(chunked):,} chunks × {chunk_len} tokens "
        f"({len(chunked) * chunk_len / 1e6:.1f}M tokens)"
    )

    def _collate(batch):
        return torch.stack([x["input_ids"] for x in batch])

    return DataLoader(
        chunked,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=_collate,
    )


# ── Pre-tokenized streaming data pipeline ────────────────────────────────────


class PreTokenizedDataset(IterableDataset):
    """Stream fixed-length chunks from pre-tokenized .npy shard files.

    Each shard is a 1-D uint16 array of concatenated, EOS-separated token IDs.
    The dataset yields (seq_len+1,) LongTensor chunks for next-token prediction.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 4096,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        self.shard_paths = sorted(glob.glob(f"{data_dir}/*.npy"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .npy shards found in {data_dir}")
        self.chunk_len = seq_len + 1
        self.shuffle_shards = shuffle_shards
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        shard_paths = list(self.shard_paths)
        if self.shuffle_shards:
            rng.shuffle(shard_paths)

        # In DDP, each rank processes a disjoint subset of shards
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            shard_paths = shard_paths[rank::world_size]

        remainder = np.array([], dtype=np.uint16)

        for path in shard_paths:
            shard = np.load(path, mmap_mode="r")

            if len(remainder) > 0:
                # Need to copy mmap data before concatenating
                combined = np.concatenate([remainder, np.array(shard)])
            else:
                combined = np.array(shard)

            n_chunks = len(combined) // self.chunk_len
            for i in range(n_chunks):
                start = i * self.chunk_len
                chunk = combined[start : start + self.chunk_len]
                yield torch.from_numpy(chunk.astype(np.int64))

            remainder = combined[n_chunks * self.chunk_len :]


def make_pretokenized_dataloader(
    data_dir: str,
    seq_len: int = 4096,
    micro_batch_size: int = 2,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    """Create a streaming DataLoader from pre-tokenized .npy shards."""
    dataset = PreTokenizedDataset(data_dir, seq_len, seed=seed)

    n_shards = len(dataset.shard_paths)
    total_bytes = sum(p.stat().st_size for p in map(Path, dataset.shard_paths))
    approx_tokens = total_bytes // 2  # uint16
    print(
        f"Pre-tokenized data: {n_shards} shards, ~{approx_tokens/1e9:.1f}B tokens, "
        f"seq_len={seq_len}, micro_batch={micro_batch_size}"
    )

    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
