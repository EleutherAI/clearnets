import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from multiprocessing import cpu_count
from safetensors.torch import load_model
from itertools import chain

import torch
from torch import Tensor
import torch.distributed as dist
from datasets import Dataset, load_dataset
from simple_parsing import field, parse
from transformers import AutoModel, BitsAndBytesConfig, PreTrainedModel, AutoProcessor, BaseImageProcessor, PreTrainedTokenizerBase, AutoTokenizer
from sparsify.data import chunk_and_tokenize, MemmapDataset
from sparsify.trainer import SaeTrainer, TrainConfig

from clearnets.train.train_transformer import LightningWrapper, MODEL_CONFIG
from clearnets.train.sparse_gptneox import SparseGPTNeoForCausalLM
from clearnets.train.sparse_gptneox_config import SparseGPTNeoConfig



# Modified from the sae __main__ - added ckpt to enable a local custom model 
@dataclass
class RunConfig(TrainConfig):
    ckpt: str = "data/roneneldan--TinyStories/Dense-TinyStories8M-s=42-full-vocab/checkpoints/last.ckpt"
    
    model: str = field(
        default="roneneldan/TinyStories-8M",
        positional=True,
    )
    """Name of the model to train."""

    dataset: str = field(
        default="roneneldan/TinyStories",
        positional=True,
    )
    """Path to the dataset to use for training."""

    split: str = "train"
    """Dataset split to use for training."""

    ctx_len: int = 512
    """Context length to use for training."""

    hf_token: str | None = None
    """Huggingface API token for downloading models."""

    revision: str | None = None
    """Model revision to use for training."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    resume: bool = False
    """Whether to try resuming from the checkpoint present at `run_name`."""

    finetune: str | None = None
    """Path to pretrained SAEs to finetune."""

    seed: int = 42
    """Random seed for shuffling the dataset."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""


def load_artifacts(args: RunConfig, rank: int) -> tuple[PreTrainedModel, Dataset | MemmapDataset]:
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
            if args.load_in_8bit
            else None
        ),
        revision=args.revision,
        torch_dtype=dtype,
        token=args.hf_token,
    )

    # For memmap-style datasets
    if args.dataset.endswith(".bin"):
        dataset = MemmapDataset(args.dataset, args.ctx_len, args.max_examples)
    else:
        dataset = load_dataset(args.dataset, split="train")
        assert isinstance(dataset, Dataset)

        processor = AutoProcessor.from_pretrained(args.model, token=args.hf_token)
        target_column = "pixel_values" if isinstance(processor, BaseImageProcessor) else "input_ids"

        if target_column not in dataset.column_names:
            if isinstance(processor, BaseImageProcessor):
                image_key = "img" if "img" in dataset.column_names else "image"    
                dataset = dataset.map(lambda x: processor(x[image_key], return_tensors="pt"), batched=True)
            elif isinstance(processor, PreTrainedTokenizerBase):
                dataset = chunk_and_tokenize(
                    dataset,
                    processor,
                    max_seq_len=args.ctx_len,
                    num_proc=args.data_preprocessing_num_proc,
                )
        else:
            print("Dataset already tokenized; skipping tokenization.")

        print(f"Shuffling dataset with seed {args.seed}")
        dataset = dataset.shuffle(args.seed)

        dataset = dataset.with_format("torch", dtype=dtype if target_column == "pixel_values" else None)
        dataset = dataset.select(list(chain(*[range(len(dataset))] * 100)))

        if limit := args.max_examples:
            dataset = dataset.select(range(limit))

    return model, dataset


# Modified from the sae __main__ to use a local custom model
def run():
    args = parse(RunConfig)

    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    # Awkward hack to prevent other ranks from duplicating data preprocessing
    if not ddp or rank == 0:
        model, dataset = load_artifacts(args, rank)
        
        # Override pretrained model with one from checkpoint
        tokenizer = AutoTokenizer.from_pretrained(args.dataset)
        model = LightningWrapper.load_from_checkpoint(
            checkpoint_path=args.ckpt,
            model=SparseGPTNeoForCausalLM(SparseGPTNeoConfig(**MODEL_CONFIG["roneneldan/TinyStories-8M"], sparse_mlp=False)),
            dense=True, 
            tokenizer=tokenizer,
            map_location=f"cuda:{rank}"
        ).model

    if ddp:
        dist.barrier()
        if rank != 0:
            model, dataset = load_artifacts(args, rank)

            # Override pretrained model with one from checkpoint
            tokenizer = AutoTokenizer.from_pretrained(args.dataset)
            model = LightningWrapper.load_from_checkpoint(
                checkpoint_path=args.ckpt,
                model=SparseGPTNeoForCausalLM(SparseGPTNeoConfig(**MODEL_CONFIG["roneneldan/TinyStories-8M"], sparse_mlp=False)),
                dense=True, 
                tokenizer=tokenizer,
                map_location=f"cuda:{rank}"
            ).model

        dataset = dataset.shard(dist.get_world_size(), rank)

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        print(f"Training on '{args.dataset}' (split '{args.split}')")
        print(f"Storing model weights in {model.dtype}")

        trainer = SaeTrainer(args, dataset, model)
        if args.resume:
            trainer.load_state(args.run_name or "sae-ckpts")
        elif args.finetune:
            for name, sae in trainer.saes.items():
                load_model(sae, f"{args.finetune}/{name}/sae.safetensors", device=str(model.device))

        trainer.fit()


if __name__ == "__main__":
    run()
