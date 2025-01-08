import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from multiprocessing import cpu_count
from safetensors.torch import load_model

import torch
from torch import Tensor
import torch.distributed as dist
from datasets import Dataset, load_dataset
from simple_parsing import field, parse
from transformers import AutoModel, BitsAndBytesConfig, PreTrainedModel, AutoProcessor, BaseImageProcessor, PreTrainedTokenizerBase, AutoTokenizer
from sae.data import chunk_and_tokenize, MemmapDataset
from sae.trainer import SaeTrainer, TrainConfig

from clearnets.sparse_feedfwd_transformer.train_tinystories_transformers import TinyStoriesModel


@dataclass
class RunConfig(TrainConfig):
    model: str = field(
        default="EleutherAI/pythia-160m",
        positional=True,
    )
    """Name of the model to train."""

    dataset: str = field(
        default="togethercomputer/RedPajama-Data-1T-Sample",
        positional=True,
    )
    """Path to the dataset to use for training."""

    split: str = "train"
    """Dataset split to use for training."""

    ctx_len: int = 2048
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


def load_tinystories_artifacts(args, rank: int) -> tuple[PreTrainedModel, Dataset | MemmapDataset, dict[str, Tensor] | None]:
    tokenizer = AutoTokenizer.from_pretrained("data/tinystories/restricted_tokenizer")
    pl_model = TinyStoriesModel.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        dense=True, 
        tokenizer=tokenizer,
        map_location=f"cuda:{rank}"
    )
    model = pl_model.model

    dataset = load_dataset("roneneldan/TinyStories", split="train")
    # max_length from TinyStories paper https://arxiv.org/pdf/2305.07759
    dataset = chunk_and_tokenize(dataset, tokenizer, max_seq_len=512) # type: ignore
    dataset.set_format(type="torch", columns=["input_ids"])

    dummy_inputs = {'input_ids': dataset[0]['input_ids'].unsqueeze(0)} 

    return model, dataset, dummy_inputs


def load_artifacts(args: RunConfig, rank: int) -> tuple[PreTrainedModel, Dataset | MemmapDataset, dict[str, Tensor] | None]:
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
    dummy_inputs = None

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

        if limit := args.max_examples:
            dataset = dataset.select(range(limit))

        dummy_inputs = {target_column: dataset[0][target_column].unsqueeze(0)} 

    return model, dataset, dummy_inputs



def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")


    args = parse(RunConfig)
    args.model = "SparseMLPTransformer"
    args.dataset = "RestrictedTinyStories"
    args.ckpt = "data/tinystories/mlp=1024-dense-8m-max-e=200-esp=15-s=42/checkpoints/last.ckpt" # type: ignore

    # Awkward hack to prevent other ranks from duplicating data preprocessing
    if not ddp or rank == 0:
        model, dataset, dummy_inputs = load_tinystories_artifacts(args, rank)
    if ddp:
        dist.barrier()
        if rank != 0:
            model, dataset, dummy_inputs = load_tinystories_artifacts(args, rank)
        dataset = dataset.shard(dist.get_world_size(), rank)

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        print(f"Training on '{args.dataset}' (split '{args.split}')")
        print(f"Storing model weights in {model.dtype}")

        trainer = SaeTrainer(args, dataset, model, dummy_inputs)
        if args.resume:
            trainer.load_state(args.run_name or "sae-ckpts")
        elif args.finetune:
            for name, sae in trainer.saes.items():
                load_model(sae, f"{args.finetune}/{name}/sae.safetensors", device=str(model.device))

        trainer.fit()


if __name__ == "__main__":
    run()
