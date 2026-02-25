"""Data loading and tokenization utilities for LoopLM pre-training."""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


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

    Args:
        max_chunks: cap the number of training chunks (useful for quick runs)
    """
    from datasets import load_dataset

    tokenizer = make_tokenizer(tokenizer_id)
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = [row[text_column] for row in dataset if row[text_column].strip()]

    chunks = tokenize_and_chunk(texts, tokenizer, seq_len)

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    print(f"Dataset: {len(chunks):,} chunks × {seq_len + 1} tokens "
          f"({len(chunks) * (seq_len + 1) / 1e6:.1f}M tokens)")

    return DataLoader(
        TensorDataset(chunks),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
