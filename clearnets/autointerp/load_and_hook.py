from functools import partial
import torch

from typing import List, Callable
from transformers import PreTrainedModel

from delphi.sparse_coders.load_sparsify import resolve_path

def hook_clearnet(
    model: PreTrainedModel,
    hookpoints: List[str],
    width: int,
    mlp_mode: str,
    device: str | torch.device | None = None,
    
) -> dict[str, Callable]:
    """
    Add a densifying function for each hookpoint since the clearnet is already sparsified.

    Args:
        model (Any): The clearnet.
        hookpoints (List[str]): List of hookpoints to adapt.
        device (str | torch.device | None, optional): The device to load the sparse models on.
            If not specified the sparse models will be loaded on the same device as the base model.

    Returns:
        dict[str, Callable]: A dictionary containing densifying functions for each hookpoint.
    """
    if device is None:
        device = model.device
    
    def to_dense(x, num_latents):
        dense = torch.zeros(
            *x['top_acts'].shape[:-1],
            num_latents,
            device=x['top_acts'].device,
            dtype=x['top_acts'].dtype,
            requires_grad=True,
        )
        dense.scatter_(-1, x['top_indices'], x['top_acts'])
        return dense
    forward = partial(to_dense, num_latents=width)

    hookpoints_to_get_sparse_acts = {}
    for hookpoint in hookpoints:
        path_segments = resolve_path(model, hookpoint.split('.'))
        if path_segments is None:
            raise ValueError(f"Could not find valid path for hookpoint: {hookpoint}")
        
        resolved_hookpoint = ".".join(path_segments)

        if 'mlp' not in resolved_hookpoint:
            hookpoints_to_get_sparse_acts[resolved_hookpoint] = lambda x: x
        else:
            hookpoints_to_get_sparse_acts[resolved_hookpoint] = forward

    return hookpoints_to_get_sparse_acts


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