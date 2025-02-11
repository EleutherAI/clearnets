from functools import reduce, partial
import torch
from typing import List, Tuple, Dict, Any

from nnsight import LanguageModel

from delphi.autoencoders.wrapper import AutoencoderLatents
from delphi.autoencoders.eleuther import resolve_path

def load_and_hook_clearnet(
    model: LanguageModel,
    hookpoints: List[str],
    device: str | torch.device | None = None
) -> Tuple[Dict[str, Any], Any]:
    """
    Load clearnet with autoencoder-adapted hookpoints.

    Args:
        model (Any): The clearnet.
        hookpoints (List[str]): List of hookpoints to adapt.
        device (str | torch.device | None, optional): The device to load the sparse models on.
            If not specified the sparse models will be loaded on the same device as the base model.

    Returns:
        Tuple[Dict[str, Any], Any]: A tuple containing the submodules dictionary and the edited model.
    """
    if device is None:
        device = model.device

    def to_dense(
        top_acts: torch.Tensor, top_indices: torch.Tensor, num_latents: int, instance_dims=[0, 1]
    ):
        """Out-of-place scatter due to apparent NNsight bug."""
        instance_shape = [top_acts.shape[i] for i in instance_dims]
        dense_empty = torch.zeros(
            *instance_shape,
            num_latents,
            device=top_acts.device,
            dtype=top_acts.dtype,
            requires_grad=True,
        )
        return dense_empty.scatter(-1, top_indices.long(), top_acts)

    
    def _forward(x, num_latents):
        return to_dense(x['top_acts'], x['top_indices'], num_latents=num_latents)

    submodules = {}
    for hookpoint in hookpoints:
        path_segments = resolve_path(model, hookpoint.split('.'))
        if path_segments is None:
            raise ValueError(f"Could not find valid path for hookpoint: {hookpoint}")
        
        submodule = reduce(getattr, path_segments, model)

        submodule.ae = AutoencoderLatents(
            torch.nn.Identity(),
            partial(_forward, num_latents=submodule.dense_h_to_4h.out_features),
            width=submodule.dense_h_to_4h.out_features # type: ignore
        )
        submodules[hookpoint] = submodule

    # Edit base model to collect sparse model activations
    with model.edit("") as edited:
        for _, submodule in submodules.items():
            submodule.ae(submodule.output)

    return submodules, edited
