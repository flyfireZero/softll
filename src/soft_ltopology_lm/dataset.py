"""
Dataset Ingestion, Pre-tokenization, and Streaming Dataloader Pipeline.
Supports:
1. WikiText-2 (Local BPE Corpus: ~2.45M tokens)
2. TinyStories (High-Coherence Narrative Benchmark: 4.6M to 125M tokens)
   - Downloads parquet shards via huggingface_hub
   - Fast tokenization and caching to .pt tensors
   - Continuous non-overlapping sequence chunking
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import pandas as pd
from huggingface_hub import hf_hub_download


class FlatTokenSequenceDataset(Dataset):
    r"""
    Fast, contiguous language modeling dataset.
    Chunks a flat 1D token tensor into non-overlapping sequences of length seq_len.
    """
    def __init__(self, token_tensor: torch.Tensor, seq_len: int = 64):
        self.seq_len = seq_len
        # Ensure token_tensor is long
        self.tokens = token_tensor.to(torch.long)
        self.num_samples = (len(self.tokens) - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


def load_or_tokenize_tinystories(
    tokenizer,
    max_stories: int = 25000,
    cache_path: str = "data/tinystories_tokens_cache.pt",
) -> torch.Tensor:
    r"""
    Loads TinyStories from cache or downloads parquet shard and tokenizes stories.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        print(f"Loading cached TinyStories tokens from {cache_path}...")
        tokens = torch.load(cache_path, weights_only=True)
        print(f"Loaded {len(tokens):,} tokens from cache.")
        return tokens

    print("Fetching TinyStories dataset shard via huggingface_hub...")
    # Use the validation shard containing 21,990 clean, multi-paragraph stories (~4.6M tokens)
    parquet_path = hf_hub_download(
        repo_id="roneneldan/TinyStories",
        filename="data/validation-00000-of-00001-869c898b519ad725.parquet",
        repo_type="dataset",
    )
    
    print(f"Reading parquet from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    if max_stories is not None and max_stories < len(df):
        df = df.iloc[:max_stories]
        
    print(f"Tokenizing {len(df):,} stories with GPT-2 BPE tokenizer...")
    all_tokens = []
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256
    
    for i, story in enumerate(df["text"]):
        toks = tokenizer.encode(story)
        all_tokens.extend(toks)
        all_tokens.append(eos_id)
        if (i + 1) % 5000 == 0:
            print(f"  Tokenized {i + 1:,}/{len(df):,} stories ({len(all_tokens):,} tokens)...")
            
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    print(f"Saving {len(tokens_tensor):,} tokens to {cache_path}...")
    torch.save(tokens_tensor, cache_path)
    return tokens_tensor


def get_tinystories_dataloader(
    tokenizer,
    seq_len: int = 64,
    batch_size: int = 16,
    max_stories: int = 25000,
    shuffle: bool = True,
) -> DataLoader:
    r"""
    Returns a PyTorch DataLoader over the TinyStories dataset.
    """
    tokens = load_or_tokenize_tinystories(tokenizer, max_stories=max_stories)
    dataset = FlatTokenSequenceDataset(tokens, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def get_tinystories_train_val_dataloaders(
    tokenizer,
    seq_len: int = 128,
    batch_size: int = 8,
    val_ratio: float = 0.05,
    max_stories: int = 25000,
) -> Tuple[DataLoader, DataLoader]:
    r"""
    Returns (train_loader, val_loader) over TinyStories with a held-out validation split.
    """
    tokens = load_or_tokenize_tinystories(tokenizer, max_stories=max_stories)
    split_idx = int(len(tokens) * (1.0 - val_ratio))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    train_dataset = FlatTokenSequenceDataset(train_tokens, seq_len=seq_len)
    val_dataset = FlatTokenSequenceDataset(val_tokens, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


def get_unified_dataloader(
    corpus_name: str,
    tokenizer,
    seq_len: int = 64,
    batch_size: int = 16,
) -> DataLoader:
    r"""
    Unified dataloader factory for 'wikitext-2' and 'tinystories'.
    """
    corpus_lower = corpus_name.lower().strip()
    if "tiny" in corpus_lower:
        return get_tinystories_dataloader(tokenizer, seq_len=seq_len, batch_size=batch_size)
    elif "wiki" in corpus_lower:
        from soft_ltopology_lm.trainer import get_wikitext_dataloader
        return get_wikitext_dataloader(tokenizer, seq_len=seq_len, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown corpus name '{corpus_name}'. Choose 'wikitext-2' or 'tinystories'.")
