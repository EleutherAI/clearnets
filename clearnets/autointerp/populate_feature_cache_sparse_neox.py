import os
from nnsight import NNsight
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
import numpy as np
import torch
import torch.distributed as dist

from clearnets.train.sparse_gptneox import SparseGPTNeoXForCausalLM

from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.config import CacheConfig

def to_dense(
    top_acts: torch.Tensor, top_indices: torch.Tensor, num_latents: int, instance_dims=[0, 1]
):
    instance_shape = [top_acts.shape[i] for i in instance_dims]
    dense_empty = torch.zeros(
        *instance_shape,
        num_latents,
        device=top_acts.device,
        dtype=top_acts.dtype,
        requires_grad=True,
    )
    return dense_empty.scatter(-1, top_indices.long(), top_acts)

def parse_args():
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--tokenizer_model", type=str, default="EleutherAI/FineWeb-restricted")
    parser.add_argument("--model_ckpt", type=str, default="/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/sparse-checkpoint-164000")
    args = parser.parse_args()
    cfg = args.options

    return cfg, args

def main():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    rank = int(os.environ.get('LOCAL_RANK') or 0)
    torch.cuda.set_device(rank)

    cfg, args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    model = SparseGPTNeoXForCausalLM.from_pretrained(
        args.model_ckpt,
        device_map={"": f"cuda:{rank}"},
    )

    model = NNsight(model, device_map="auto", torch_dtype=torch.bfloat16, tokenizer=tokenizer)
    model.tokenizer = tokenizer

    submodule_dict = {}

    for layer in range(len(model.gpt_neox.layers)):
        def _forward(x):
            return to_dense(x['top_acts'], x['top_indices'], num_latents=512 * 4 * 8)

        submodule = model.gpt_neox.layers[layer].mlp
        submodule.ae = AutoencoderLatents(
            None, _forward, width=512 * 4 * 8
        )
        submodule_dict[submodule.path] = submodule
    
    with model.edit("") as edited:
        for _, submodule in submodule_dict.items():
            acts = submodule.output
            submodule.ae(acts, hook=True)

    tokens = load_tokenized_data(
            ctx_len=512,
            tokenizer=tokenizer,
            dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
            dataset_split="train[:1%]",
            dataset_row="text"
    )

    cache = FeatureCache(
        edited,
        submodule_dict,
        batch_size = 48,
    )

    cache.run(n_tokens = 10_000_000, tokens=tokens)

    save_dir = "/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/cached_activations/sparse"

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=save_dir
    )
    cache.save_config(
        save_dir=save_dir,
        cfg=cfg,
        model_name=args.tokenizer_model
    )

if __name__ == "__main__":
    main()