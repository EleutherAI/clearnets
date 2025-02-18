from pathlib import Path
from glob import glob
import json
import torch
import asyncio
import time
import plotly.express as px

from transformers import AutoConfig
from dataclasses import dataclass
from simple_parsing import field
from delphi.config import ExperimentConfig, LatentConfig, CacheConfig
from delphi.log.result_analysis import build_scores_df, latent_balanced_score_metrics
from delphi.__main__ import RunConfig, process_cache, populate_cache
from delphi.sparse_coders import load_sparsify_sparse_coders
from transformers import BitsAndBytesConfig, AutoTokenizer

from clearnets.train.sparse_gptneox import SparseGPTNeoXForCausalLM

from clearnets.autointerp.load_and_hook import hook_clearnet

@dataclass
class CustomModelRunConfig:
    tokenizer_model: str = field(
        default="EleutherAI/FineWeb-restricted",
        positional=True,
    )
    mlp_mode: str = field(
        default="dense",
        positional=True,
        options=["dense", "sparse", "sparse_low_rank", "sparse_group_max"],
    )


def load_artifacts(run_cfg: RunConfig, custom_run_cfg: CustomModelRunConfig):
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    tokenizer = AutoTokenizer.from_pretrained(custom_run_cfg.tokenizer_model)
    if Path(run_cfg.model).exists():
        model_cfg = AutoConfig.from_pretrained(Path(run_cfg.model) / "config.json")
        model_cfg.mlp_mode = custom_run_cfg.mlp_mode # type: ignore lol
    else:
        model_cfg = None

    model = SparseGPTNeoXForCausalLM.from_pretrained(
        run_cfg.model,
        config=model_cfg,
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=run_cfg.hf_token,
    )

    if run_cfg.sparse_model == "":
        width = 0
        if custom_run_cfg.mlp_mode == "sparse_low_rank":
            width = model.gpt_neox.layers[0].mlp.encoder[1].out_features
        elif custom_run_cfg.mlp_mode == "sparse_group_max":
            raise NotImplementedError("Sparse group max is not implemented for the clearnet")
        elif custom_run_cfg.mlp_mode == "sparse":
            width = model.gpt_neox.layers[0].mlp.dense_h_to_4h.out_features

        hookpoints_to_get_sparse_acts = hook_clearnet(
            model,
            run_cfg.hookpoints,
            width=width,
            mlp_mode=custom_run_cfg.mlp_mode,
            
        )
    else:
        hookpoints_to_get_sparse_acts = load_sparsify_sparse_coders(
            model,
            run_cfg.sparse_model,
            hookpoints=run_cfg.hookpoints,
            compile=True
        )

    return hookpoints_to_get_sparse_acts, model, tokenizer


def all_configs():
    for model, sparse_model, mlp_mode in [
        (
            # Warning: this model was produced when the config used sparse_mlp: bool rather than mlp_mode: str and it 
            # currently doesn't load properly
            "/mnt/ssd-1/nora/sparse-run/HuggingFaceFW--fineweb/Sparse-FineWeb10B-28M-s=42/checkpoints/checkpoint-57280",
            "",
            "sparse",
        ),
        (
            "/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/sparse-checkpoint-164000",
            "",
            "sparse",
        ),
        (
            "/mnt/ssd-1/nora/dense-ckpts/checkpoint-118000",
            "/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/sae_8x",
            "",
        ),
    ]:
        pass


async def test_clearnet():

    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:1%]",
        dataset_column="text",
        batch_size=8,
    )
    experiment_cfg = ExperimentConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=40,
        n_examples_test=50,
    )
    overwrite = []
    overwrite.append("cache")
    overwrite.append("scores")

    run_cfg = RunConfig(
        name='clearnet-164000',
        overwrite=overwrite,
        model="/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/sparse-checkpoint-164000",
        # model="/mnt/ssd-1/nora/dense-ckpts/checkpoint-118000",
        # sparse_model="/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/sae_8x",
        sparse_model="",
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        hookpoints=["gpt_neox.layers.5.mlp"],
        explainer_model_max_len=4208,
        max_latents=100,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True,
    )
    custom_run_cfg = CustomModelRunConfig(
        tokenizer_model="EleutherAI/FineWeb-restricted",
        mlp_mode="sparse",
    )

    base_path = Path.cwd() / "results"
    if run_cfg.name:
        base_path = base_path / run_cfg.name

    base_path.mkdir(parents=True, exist_ok=True)
    
    with open(base_path / "custom_run_config.json", "w") as f:
        json.dump(custom_run_cfg.__dict__, f, indent=4)

    with open(base_path / "run_config.json", "w") as f:
        json.dump(run_cfg.__dict__, f, indent=4)

    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"
    visualize_path = base_path / "visualize"
    visualize_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    feature_range = (
        torch.arange(run_cfg.max_latents) if run_cfg.max_latents else None
    )
    hookpoints_to_sparse_encode, model, tokenizer = load_artifacts(
        run_cfg, custom_run_cfg
    )

    latent_cfg = LatentConfig(
        min_examples=200,
        max_examples=10_000,
    )

    if (
        not glob(str(latents_path / ".*")) + glob(str(latents_path / "*"))
        or "cache" in run_cfg.overwrite
    ):
        populate_cache(
            run_cfg,
            cache_cfg,
            experiment_cfg,
            model,
            hookpoints_to_sparse_encode,
            latents_path,
            tokenizer,
        )
    else:
        print(f"Files found in {latents_path}, skipping cache population...")

    resolved_hookpoints = list(hookpoints_to_sparse_encode.keys())
    
    del model, hookpoints_to_sparse_encode

    if (
        not glob(str(scores_path / ".*")) + glob(str(scores_path / "*"))
        or "scores" in run_cfg.overwrite
    ):
        await process_cache(
            latent_cfg,
            run_cfg,
            experiment_cfg,
            latents_path,
            explanations_path,
            scores_path,
            resolved_hookpoints,
            tokenizer,
            feature_range,
        )
    else:
        print(f"Files found in {scores_path}, skipping...")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    scores_path =  Path("results") / run_cfg.name / "scores"

    df = build_scores_df(scores_path, resolved_hookpoints, feature_range)
    

    for score_type in df["score_type"].unique():
        score_df = df[df['score_type'] == score_type]
        latent_balanced_score_metrics(score_df, score_type)

        fig = px.histogram(
            df[df["score_type"] == score_type],
            x="accuracy",
            barmode="overlay",
            title=f"Accuracy distribution - {score_type} (run: {run_cfg.name})",
            nbins=100,
        )
        fig.write_image(visualize_path / f"autointerp_accuracies_{score_type}.pdf", format="pdf")



if __name__ == "__main__":
    asyncio.run(test_clearnet())