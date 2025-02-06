import os

from nnsight import NNsight
from simple_parsing import ArgumentParser
import torch
from delphi.config import CacheConfig
from transformers import AutoTokenizer

from clearnets.autointerp.autointerp_load_saes import load_eai_autoencoders
from clearnets.autointerp.populate_feature_cache_sparse import save_features
from clearnets.train.sparse_gptneox import SparseGPTNeoXForCausalLM


def load_artifacts(ckpt_path: str, sae_dir):

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/FineWeb-restricted")
    model = SparseGPTNeoXForCausalLM.from_pretrained(
        ckpt_path,
        # device_map={"": f"cuda:{rank}"},
        torch_dtype=torch.bfloat16
    )

    model.to('cuda') # type: ignore
    nnsight_model = NNsight(model, device_map="auto", torch_dtype=torch.bfloat16, tokenizer=tokenizer)
    nnsight_model.tokenizer = tokenizer

    # Modified from the original helper function in sae-auto-interp to be compatible with TinyStories hookpoint naming:
    # submodule = f"transformer.h.{layer}.mlp"
    # submodule = model.transformer.h[layer].mlp
    submodule_dict, nnsight_model = load_eai_autoencoders(
        nnsight_model,
        list(range(len(model.gpt_neox.layers))),
        sae_dir,
        module="mlp",
    )

    return nnsight_model, submodule_dict


def parse_args():
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--model_ckpt", type=str, default="/mnt/ssd-1/nora/dense-ckpts/checkpoint-118000"),
    parser.add_argument("--tokenizer_model", type=str, default="EleutherAI/FineWeb-restricted", help="Model class to load the tokenizer for")
    parser.add_argument("--dataset", type=str, default="EleutherAI/fineweb-edu-dedup-10b")
    parser.add_argument("--sae_dir", type=str, default="/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/transcoder")
    parser.add_argument("--save_dir", type=str, default="/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/cached_activations/transcoder")
    return parser.parse_args()


def main(): 
    args = parse_args()
    cfg = args.options

    os.makedirs(args.save_dir, exist_ok=True)
    model, submodule_dict = load_artifacts(args.model_ckpt, args.sae_dir)
    
    save_features(cfg, model, submodule_dict, args.save_dir, args)
    
    print(f"Saved feature splits and config to {args.save_dir}")


if __name__ == "__main__":
    main()