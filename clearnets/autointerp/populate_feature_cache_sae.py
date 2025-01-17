import os

from nnsight import NNsight
from simple_parsing import ArgumentParser
import torch
from sae_auto_interp.config import CacheConfig
from transformers import AutoTokenizer

from clearnets.autointerp.autointerp_load_saes import load_eai_autoencoders
from clearnets.autointerp.populate_feature_cache_sparse import save_features
from clearnets.train.sparse_gptneox import SparseGPTNeoForCausalLM
from clearnets.train.sparse_gptneox_config import SparseGPTNeoConfig
from clearnets.train.train_transformer import LightningWrapper, MODEL_CONFIG


def load_artifacts(cfg, ckpt_path: str, dataset: str, sae_dir, features_name):
    save_dir = f"data/raw_features/{cfg.dataset_repo}/{features_name}"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(dataset)
    model = LightningWrapper.load_from_checkpoint(
        ckpt_path,
        model=SparseGPTNeoForCausalLM(SparseGPTNeoConfig(**MODEL_CONFIG["roneneldan/TinyStories-8M"], sparse_mlp=False)),
        dense=True,
        tokenizer=tokenizer
    ).model

    model.to('cuda') # type: ignore
    nnsight_model = NNsight(model, device_map="auto", torch_dtype=torch.bfloat16, tokenizer=tokenizer)
    nnsight_model.tokenizer = tokenizer

    # Modified from the original helper function in sae-auto-interp to be compatible with TinyStories hookpoint naming:
    # submodule = f"transformer.h.{layer}.mlp"
    # submodule = model.transformer.h[layer].mlp
    submodule_dict, nnsight_model = load_eai_autoencoders(
        nnsight_model,
        list(range(len(model.transformer.h))),
        sae_dir,
        module="mlp",
    )

    return nnsight_model, save_dir, submodule_dict


def parse_args():
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--tag", type=str, help="Tag for naming the cache directory")
    parser.add_argument("--tokenizer_model", type=str, default="roneneldan/TinyStories-8M", help="Model class to load the tokenizer for")
    parser.add_argument("--model_cfg", type=str, default="roneneldan/TinyStories-8M", help="Tag to load a custom config")
    parser.add_argument("--model_ckpt", type=str, default="data/roneneldan--TinyStories/Dense-TinyStories8m-s=42/checkpoints/last.ckpt")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--sae_dir", type=str, default="/mnt/ssd-1/lucia/clearnets/data/sae/Dense TinyStories8M Transcoder 32x 8192 s=42 epoch 21")
    return parser.parse_args()


def main(): 
    args = parse_args()
    cfg = args.options

    model, save_dir, submodule_dict = load_artifacts(cfg, args.model_ckpt, args.dataset, args.sae_dir, args.tag)
    
    save_features(cfg, model, submodule_dict, save_dir, args)
    
    print(f"Saved feature splits and config to {save_dir}")


if __name__ == "__main__":
    main()