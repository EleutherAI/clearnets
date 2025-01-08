# Functions to collect SAE-based metrics

from typing import Any
from collections import defaultdict

import torch
import numpy as np
from torch import Tensor
from sae.sae import Sae
from torch.utils.data import DataLoader

from clearnets.metrics import gini, hoyer, hoyer_square, abs_entropy
from clearnets.pgdl.complexity_measures import get_generalization_score


def get_batch_metrics(
    acts: Tensor,
    accumulator: defaultdict[str, list],
    feature_dims=[2],
    instance_dims=[0, 1],
):
    acts = acts.to(torch.float64)
    if acts.ndim != 2:
        permuted_dims = instance_dims + feature_dims
        acts = acts.permute(*permuted_dims)
        acts = acts.flatten(0, len(instance_dims) - 1)
        acts = acts.flatten(1, len(feature_dims))

    l2_norms = torch.norm(acts, dim=1)

    # L2 norm mean
    accumulator["mean_l2"].append(l2_norms.mean().item())

    # L2 norm variance
    accumulator["var_l2"].append(l2_norms.var().item())

    # L2 norm skewness: E[(X - μ)³] / σ³
    centered = l2_norms - l2_norms.mean()
    skew = (centered**3).mean() / (l2_norms.std() ** 3)
    accumulator["skew_l2"].append(skew.item())

    # L2 norm kurtosis: E[(X - μ)⁴] / σ⁴ - 3 (subtracting 3 gives excess kurtosis)
    kurt = (centered**4).mean() / (l2_norms.var() ** 2) - 3
    accumulator["kurt_l2"].append(kurt.item())

    # Element-wise variance
    acts_centered = acts - acts.mean(dim=0, keepdim=True)
    var_trace = (acts_centered.pow(2)).mean(dim=0).sum().item()
    accumulator["var_trace"].append(var_trace)

    return accumulator


def to_dense(
    top_acts: Tensor, top_indices: Tensor, num_latents: int, instance_dims=[0, 1]
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


def concatenate_values(dictionary: dict[Any, Tensor]) -> np.ndarray:
    scores = []
    for values in dictionary.values():
        scores.extend(values.flatten().tolist())

    return np.array(scores)


def flatten_acts(acts: Tensor, instance_dims: list[int], feature_dims: list[int]):
    permuted_dims = instance_dims + feature_dims
    acts = acts.permute(*permuted_dims)
    acts = acts.flatten(0, len(instance_dims) - 1)
    acts = acts.flatten(1, len(feature_dims))

    return acts


@torch.no_grad()
def get_mean_sae_acts(
    model,
    ordered_submods: list,
    dictionaries: dict[Any, Sae],
    dataloader: DataLoader,
    device: str | torch.device = torch.device("cuda"),
    input_key: str = "input_ids",
    feature_dims=[2],
    instance_dims=[0, 1],
):
    """Meaned over a large dataset"""
    # Collect metadata
    is_tuple = {}
    first_batch = next(iter(dataloader))
    with model.scan(first_batch[input_key].to(device)):
        for submodule in ordered_submods:
            output = (
                submodule.nns_output
                if hasattr(submodule, "nns_output")
                else submodule.output
            )
            is_tuple[submodule] = type(output.shape) == tuple

    accumulated_nodes = {}
    for submodule in ordered_submods:
        num_latents = dictionaries[submodule].num_latents
        accumulated_nodes[submodule.path] = torch.zeros(num_latents, dtype=torch.float64)  # type: ignore

    # Do inference
    accumulated_fvu = {submodule.path: 0.0 for submodule in ordered_submods}
    accumulated_multi_topk_fvu = {submodule.path: 0.0 for submodule in ordered_submods}
    accumulated_sample_count = {submodule.path: 0.0 for submodule in ordered_submods}
    num_batches = 0

    with torch.amp.autocast(str(device)):
        for i, batch in enumerate(dataloader):
            inputs = {}
            with model.trace(batch[input_key].to(device)):
                for submodule in ordered_submods:
                    if hasattr(submodule, "nns_output"):
                        input = (
                            submodule.nns_output
                            if not is_tuple[submodule]
                            else submodule.nns_output[0]
                        )
                    else:
                        input = (
                            submodule.output
                            if not is_tuple[submodule]
                            else submodule.output[0]
                        )

                    inputs[submodule] = input.save()

            for submodule, input in inputs.items():
                permuted_dims = instance_dims + feature_dims
                sae_input = input.permute(*permuted_dims)
                sae_input = sae_input.flatten(0, len(instance_dims) - 1)
                sae_input = sae_input.flatten(1, len(feature_dims))

                flat_f = dictionaries[submodule].forward(sae_input)

                num_latents = dictionaries[submodule].num_latents
                num_instances = 1
                for i in instance_dims:
                    num_instances *= sae_input.shape[i]

                mean_acts = torch.zeros(
                    num_instances,
                    num_latents,
                    device=flat_f.latent_acts.device,
                    dtype=flat_f.latent_acts.dtype,
                    requires_grad=True,
                )
                mean_acts.scatter_(-1, flat_f.latent_indices.long(), flat_f.latent_acts)

                mean_acts = mean_acts.mean(0).to(torch.float64).cpu()  # type: ignore

                accumulated_nodes[submodule.path] += mean_acts  # type: ignore
                accumulated_fvu[submodule.path] += flat_f.fvu.cpu().item()  # type: ignore
                accumulated_multi_topk_fvu[submodule.path] += flat_f.multi_topk_fvu.cpu().item()  # type: ignore
                accumulated_sample_count[submodule.path] += sae_input.shape[0]

            num_batches += 1

    nodes = {
        k: v / sample_count
        for (k, v), sample_count in zip(
            accumulated_nodes.items(), accumulated_sample_count.values()
        )
    }
    fvu = {k: v / num_batches for k, v in accumulated_fvu.items()}
    multi_topk_fvu = {k: v / num_batches for k, v in accumulated_multi_topk_fvu.items()}

    return nodes, fvu, multi_topk_fvu


@torch.no_grad()
def stream_sae_acts(
    model,
    ordered_submods: list,
    dictionaries: dict[Any, Sae],
    dataloader: DataLoader,
    device: str | torch.device = torch.device("cuda"),
    input_key: str = "input_ids",
    feature_dims=[2],
    instance_dims=[0, 1],
    collect=True,
):
    """Meaned over a large dataset"""
    is_tuple = {}
    first_batch = next(iter(dataloader))
    batch_size = first_batch[input_key].shape[0]
    accumulated_nodes = defaultdict(list)
    with model.scan(first_batch[input_key].to(device)):
        for submodule in ordered_submods:
            output = (
                submodule.nns_output
                if hasattr(submodule, "nns_output")
                else submodule.output
            )
            is_tuple[submodule] = type(output.shape) == tuple

    accumulated_fvu = defaultdict(list)
    accumulated_multi_topk_fvu = defaultdict(list)
    metrics = defaultdict(list)

    with torch.amp.autocast(str(device)):
        for batch in dataloader:
            inputs = {}
            with model.trace(batch[input_key].to(device)):
                for submodule in ordered_submods:
                    if hasattr(submodule, "nns_output"):
                        input = (
                            submodule.nns_output
                            if not is_tuple[submodule]
                            else submodule.nns_output[0]
                        )
                    else:
                        input = (
                            submodule.output
                            if not is_tuple[submodule]
                            else submodule.output[0]
                        )
                    inputs[submodule] = input.save()

            for submodule, input in inputs.items():
                permuted_dims = instance_dims + feature_dims
                sae_input = input.permute(*permuted_dims)
                sae_input = sae_input.flatten(0, len(instance_dims) - 1)
                sae_input = sae_input.flatten(1, len(feature_dims))

                flat_f = dictionaries[submodule].forward(sae_input)

                accumulated_fvu[submodule.path].append(flat_f.fvu.cpu().item())
                accumulated_multi_topk_fvu[submodule.path].append(
                    flat_f.multi_topk_fvu.cpu().item()
                )

                original_instance_shape = [input.shape[i] for i in instance_dims]
                latent_acts = flat_f.latent_acts.view(*original_instance_shape, -1)
                latent_indices = flat_f.latent_indices.view(
                    *original_instance_shape, -1
                )
                dense = to_dense(
                    latent_acts,  # type: ignore
                    latent_indices,  # type: ignore
                    dictionaries[submodule].num_latents,
                    instance_dims=instance_dims,
                )

                accumulated_nodes[submodule.path].append(dense.cpu().double())  # type: ignore

            # ultra long vector with all sae features
            flattened_acts = {
                submodule: flatten_acts(
                    accumulated_nodes[submodule.path][-1],
                    instance_dims,
                    (accumulated_nodes[submodule.path][-1].ndim - 1,), # type: ignore
                )
                for submodule in ordered_submods
            }
            # Handle Swin edge case where batch size is not preserved, by folding the excess batch size
            # into the instance dimension
            for key, flattened_act in flattened_acts.items():
                if flattened_act.shape[0] != batch_size:
                    n = flattened_act.shape[0] // batch_size
                    feature_len = flattened_act.shape[-1]
                    flattened_acts[key] = flattened_act.view(
                        batch_size, n, feature_len
                    ).view(batch_size, feature_len * n)

            acts = torch.cat([a for a in flattened_acts.values()], dim=1)
            metrics = get_batch_metrics(
                acts, metrics, feature_dims=feature_dims, instance_dims=instance_dims
            )

            if not collect:
                for submodule in ordered_submods:
                    accumulated_nodes[submodule.path] = []

    mean_metrics = {}
    for key, value in metrics.items():
        mean_metrics[key] = np.mean(value)

    return accumulated_nodes, accumulated_fvu, accumulated_multi_topk_fvu, mean_metrics


def get_sae_metrics(
    model,
    nnsight_model,
    dictionaries: dict,
    submods: list,
    dataloaders: dict[str, DataLoader],
    feature_dims=(1,),
    instance_dims=(0,),
    device=torch.device("cuda"),
):
    metrics: dict[str, Any] = {}
    activations: dict[str, Any] = {}

    metrics["parameter_norm"] = (
        torch.cat([parameters.flatten() for parameters in model.parameters()])
        .flatten()
        .norm(p=2)
        .item()
    )

    for dataloader_name, dataloader in dataloaders.items():
        dataloader_metrics = {}

        dataloader_metrics['pgdl_score'] = get_generalization_score(model, dataloader)

        mean_acts, fvu, multi_topk_fvu = get_mean_sae_acts(
            nnsight_model,
            submods,
            dictionaries,
            dataloader,
            input_key="input_ids",
            feature_dims=feature_dims,
            instance_dims=instance_dims,
            device=device,
        )
        mean_acts_vec = concatenate_values(mean_acts)

        mean_fvu = np.mean([v for v in fvu.values()])
        mean_multi_topk_fvu = np.mean([v for v in multi_topk_fvu.values()])

        activations[dataloader_name] = mean_acts

        dataloader_metrics["sae_fvu"] = mean_fvu
        dataloader_metrics["sae_multi_topk_fvu"] = mean_multi_topk_fvu

        dataloader_metrics["sae_entropy"] = abs_entropy(mean_acts_vec)
        dataloader_metrics["hoyer"] = hoyer(mean_acts_vec)
        dataloader_metrics["hoyer_square"] = hoyer_square(mean_acts_vec)
        dataloader_metrics['gini'] = gini(mean_acts_vec)

        _, _, _, batch_metrics = stream_sae_acts(
            nnsight_model,
            submods,
            dictionaries,
            dataloader,
            input_key="input_ids",
            feature_dims=feature_dims,
            instance_dims=instance_dims,
            device=device,
            collect=False,
        )

        dataloader_metrics["mean_l2"] = batch_metrics["mean_l2"]
        dataloader_metrics["var_l2"] = batch_metrics["var_l2"]
        dataloader_metrics["skew_l2"] = batch_metrics["skew_l2"]
        dataloader_metrics["kurt_l2"] = batch_metrics["kurt_l2"]
        dataloader_metrics["var_trace"] = batch_metrics["var_trace"]

        metrics[dataloader_name] = dataloader_metrics

    return metrics, activations
