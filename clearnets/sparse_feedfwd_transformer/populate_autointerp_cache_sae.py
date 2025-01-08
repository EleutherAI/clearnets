import os

from nnsight import NNsight
from simple_parsing import ArgumentParser
import torch
from sae_auto_interp.config import CacheConfig
from transformers import AutoTokenizer
from clearnets.sparse_feedfwd_transformer.train_tinystories_transformers import TinyStoriesModel
from clearnets.sparse_feedfwd_transformer.autointerp_load_saes import load_eai_autoencoders
from clearnets.sparse_feedfwd_transformer.populate_autointerp_cache_sparse import save_features


def load_artifacts(cfg, model_name, sae_dir, features_name):
    save_dir = f"raw_features/{cfg.dataset_repo}/{features_name}"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("data/tinystories/restricted_tokenizer")
    ckpt_path = f'data/tinystories/{model_name}/checkpoints/last.ckpt'
    model = TinyStoriesModel.load_from_checkpoint(
        ckpt_path,
        dense=True,
        tokenizer=tokenizer
    ).model

    model.to('cuda') # type: ignore
    nnsight_model = NNsight(model, device_map="auto", torch_dtype=torch.bfloat16, tokenizer=tokenizer)
    nnsight_model.tokenizer = tokenizer

    # Hacked in two places:
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
    parser.add_argument("--out", type=str, default="Transcoder-8M")
    parser.add_argument("--model", type=str, default="mlp=1024-dense-8m-max-e=200-esp=15-s=42")
    parser.add_argument("--sae_dir", type=str, default="/mnt/ssd-1/lucia/clearnets/data/sae/Dense TinyStories8M Transcoder 32x 8192 s=42 epoch 21")
    return parser.parse_args()


def main(): 
    args = parse_args()
    cfg = args.options

    model, save_dir, submodule_dict = load_artifacts(cfg, args.model, args.sae_dir, args.out)
    
    save_features(cfg, model, submodule_dict, save_dir)
    
    print(f"Saved feature splits and config to {save_dir}")


if __name__ == "__main__":
    main()