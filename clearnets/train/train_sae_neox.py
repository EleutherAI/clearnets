import os
from pathlib import Path
from sae.config import SaeConfig, TrainConfig
from sae.trainer import SaeTrainer
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset
from sae.data import chunk_and_tokenize
import torch
import torch.distributed as dist

from clearnets.train.sparse_gptneox import SparseGPTNeoXForCausalLM

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--distribute_modules", action="store_true")
    parser.add_argument("--transcode", action="store_true")
    parser.add_argument("--batch_size", type=int, default=784)
    return parser.parse_args()

def main():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    rank = int(os.environ.get('LOCAL_RANK') or 0)
    torch.cuda.set_device(rank)

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/FineWeb-restricted")
    model = SparseGPTNeoXForCausalLM.from_pretrained(
        "/mnt/ssd-1/nora/dense-ckpts/checkpoint-118000",
        device_map={"": f"cuda:{rank}"},
        torch_dtype=torch.bfloat16
    )
    
    wandb_name = "Sparse FineWebEduDeduped 58M s=42 step 118000"

    if args.transcode:
        wandb_name += " transcode"

    dataset = load_dataset("EleutherAI/fineweb-edu-dedup-10b", split="train", cache_dir="/mnt/ssd-1/caleb/hf_cache", num_proc=64)
    dataset = chunk_and_tokenize(dataset, tokenizer, max_seq_len=512, text_key="text")
    dataset = dataset.shard(dist.get_world_size(), rank)

    cfg = TrainConfig(
        SaeConfig(expansion_factor=8, k=32),
        batch_size=args.batch_size,
        run_name=str(Path('sae') / wandb_name),
        log_to_wandb=False,
        transcode=args.transcode,
        distribute_modules=args.distribute_modules
    )
    trainer = SaeTrainer(cfg, dataset, model)
    trainer.fit()

if __name__ == "__main__":
    main()