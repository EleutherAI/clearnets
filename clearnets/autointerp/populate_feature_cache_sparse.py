import os
import glob

from nnsight import NNsight
from simple_parsing import ArgumentParser
import torch
from transformers import AutoTokenizer
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
from typing import Any, Tuple, Dict

from clearnets.train.train_tinystories_transformers import TinyStoriesModel
from clearnets.generalization.inference.inference import to_dense


def load_dense_mlp_transformer_saes(model):
    pass

def load_sparse_mlp_transformer_latents(
    model: Any,
) -> Tuple[Dict[str, Any], Any]:
    """
    Load hidden activations for specified layers and module.

    Args:
        model (Any): The model to load autoencoders for.
        layers (List[int]): List of layer indices to load autoencoders for.
        module (str): Module name ('mlp' or 'res').
        randomize (bool, optional): Whether to randomize the autoencoder. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[Dict[str, Any], Any]: A tuple containing the submodules dictionary and the edited model.
    """
    # This doesn't matter because we don't use the width
    hook_to_d_in = resolve_widths(model, get_gptneo_hookpoints(model), torch.randint(0, 10000, (1, 1024)))

    submodules = {}

    for layer in range(len(model.transformer.h)):
        def _forward(x):
            return to_dense(x['top_acts'], x['top_indices'], num_latents=256 * 4 * 8) # hook_to_d_in[hookpoint])
            # return (x['top_indices'], x['top_acts'])

        hookpoint = f"transformer.h.{layer}.mlp"
        submodule = model.transformer.h[layer].mlp
        submodule.ae = AutoencoderLatents(
            None, _forward, width=256 * 4 * 8 # hook_to_d_in[hookpoint] # type: ignore
        )
        submodules[submodule.path] = submodule


    with model.edit("") as edited:
        for path, submodule in submodules.items():
            if "embed" not in path and "mlp" not in path:
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules, edited

def resolve_widths(
    model, module_names: list[str], inputs: torch.Tensor, dim = -1
) -> dict[str, int]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = {
        model.get_submodule(name): name for name in module_names
    }
    shapes: dict[str, int] = {}

    def hook(module, _, output):
        # Unpack tuples if needed
        if isinstance(output, tuple):
            output, *_ = output

        if isinstance(output, dict):
            output = output['hidden_states']

        name = module_to_name[module]

        shapes[name] = output.shape[dim]

    handles = [
        mod.register_forward_hook(hook) for mod in module_to_name
    ]
    dummy = inputs.to(model.device)
    try:
        model._model(dummy)
    finally:
        for handle in handles:
            handle.remove()
    
    return shapes

def get_gptneo_hookpoints(model):
    hookpoints = []
    for i in range(len(model.transformer.h)):
        hookpoints.append(f"transformer.h.{i}.attn.attention")
        hookpoints.append(f"transformer.h.{i}.mlp")
    return hookpoints

@torch.inference_mode()
def main(cfg: CacheConfig, args): 
    dataset_str = "roneneldan/TinyStories"

    features_name = f"{args.model}-{args.epoch}"
    save_dir = f"raw_features/{cfg.dataset_repo}/{features_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(dataset_str)

    ckpt_pattern = f"data/{dataset_str.replace('/', '--')}/{args.model}/checkpoints/epoch={args.epoch}-step=*.ckpt"
    matching_ckpt = glob.glob(ckpt_pattern)[0]
    model = TinyStoriesModel.load_from_checkpoint(
        matching_ckpt,
        dense=False,
        tokenizer=tokenizer
    ).model
    model.to(device='cuda') # type: ignore

    # I believe dispatch won't work for tinystories models
    model = NNsight(model, device_map="auto", torch_dtype=torch.bfloat16, tokenizer=tokenizer) # dispatch=False
    model.tokenizer = tokenizer
    submodule_dict, model = load_sparse_mlp_transformer_latents(model)

    save_features(cfg, model, submodule_dict, save_dir)
    print(f"Saved feature splits and config to {save_dir}")


def save_features(cfg, model, submodule_dict, save_dir):
    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
        cfg.dataset_name,
        cfg.dataset_row,
    )
    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=cfg.batch_size,
    )
    cache.run(cfg.n_tokens, tokens) # type: ignore

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=save_dir
    )
    cache.save_config(
        save_dir=save_dir,
        cfg=cfg,
        model_name=args.model
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--model", type=str, default="sparse-8m-max-e=200-esp=15-s=42")
    # epoch=6-step=1456.ckpt has matched val loss with dense-8m-max-e=200-esp=15-s=42 epoch=21-step=1456.ckpt
    parser.add_argument("--epoch", type=int, default=6) 
    args = parser.parse_args()
    cfg = args.options
    
    main(cfg, args)