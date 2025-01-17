from typing import Optional
import os
from copy import deepcopy
from argparse import ArgumentParser

from tokenizers import models
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from tokengrams import MemmapIndex, tokenize_hf_dataset


def get_vocab_restricted_tokenizer(
    base_tokenizer_name: str,
    max_vocab_size: int = 10_000,
    save_path: Optional[str] = None,
    dataset_str: str = "roneneldan/TinyStories",
    text_key: str = "text",
):
    """
    Restrict a GPT-NeoX tokenizer's vocabulary to the most common tokens in the TinyStories dataset while preserving
    whitespace handling and token uniqueness.

    The original tokenizer uses <|endoftext|> for every special token.
    Ideally we'd handle this implicitly but it's currently hard-coded.

    Args:
        base_tokenizer: The GPT-NeoX tokenizer to restrict
        dataset: HuggingFace dataset containing text data
        max_vocab_size: Maximum number of tokens to keep
        save_path: Optional path to save the modified tokenizer
    """
    pretty_dataset_str = dataset_str.split('/')[-1]

    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    base_tokenizer.pad_token = base_tokenizer.eos_token

    original_vocab = base_tokenizer.get_vocab()
    original_tokens = {v: k for k, v in original_vocab.items()}

    # Count token frequencies
    if not os.path.exists(f"data/{pretty_dataset_str}.bin"):
        print("Tokenizing dataset...")
        tokenize_hf_dataset(
            dataset=load_dataset(dataset_str, split="train"),
            tokenizer=base_tokenizer,
            output_path=f"data/{pretty_dataset_str}.bin",
            text_key=text_key,
            append_eod=True,
            workers=10,
        )
        index = MemmapIndex.build(f"data/{pretty_dataset_str}.bin", f"data/{pretty_dataset_str}.idx")
    else:
        print("Loading index...")
        index = MemmapIndex(f"data/{pretty_dataset_str}.bin", f"data/{pretty_dataset_str}.idx")

    print("Counting tokens...")
    token_counts = index.count_next([])

    # Get most common token IDs
    most_common_token_ids = torch.topk(
        torch.tensor(token_counts), k=max_vocab_size
    ).indices.tolist()

    # Create new vocabulary preserving original token strings
    new_vocab = {}
    current_id = 0

    # First add special tokens
    special_token = "<|endoftext|>"
    new_vocab[special_token] = current_id
    current_id += 1

    for token_id in most_common_token_ids:
        if token_id < len(original_tokens):
            token_string = original_tokens[token_id]
            if token_string not in new_vocab and token_string != special_token:
                new_vocab[token_string] = current_id
                current_id += 1

    # Create tokenizer with new vocab and backend mimicking the original
    tokenizer_backend = deepcopy(base_tokenizer.backend_tokenizer)
    tokenizer_config = tokenizer_backend.model # type: ignore
    tokenizer_backend.model = models.BPE( # type: ignore
        vocab=new_vocab,
        merges=[],
        dropout=tokenizer_config.dropout,
        unk_token=base_tokenizer.unk_token,
        continuing_subword_prefix=tokenizer_config.continuing_subword_prefix,
        end_of_word_suffix=tokenizer_config.end_of_word_suffix,
        fuse_unk=tokenizer_config.fuse_unk,
    )
    new_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        trim_offsets=True,  # Changed to True to better handle whitespace
    )

    if save_path:
        new_tokenizer.save_pretrained(save_path)

    return new_tokenizer


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--dataset", type=str, default="lennart-finke/SimpleStories")
    args = args.parse_args()

    pretty_dataset_str = args.dataset.split('/')[-1]

    tokenizer = get_vocab_restricted_tokenizer(
        "EleutherAI/gpt-neo-2.7B", 
        save_path=f"data/{pretty_dataset_str}/restricted_tokenizer",
        dataset_str=args.dataset,
        text_key="story"
    )
    tokenizer.push_to_hub(f"EleutherAI/{pretty_dataset_str}-restricted")

    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{pretty_dataset_str}-restricted")
