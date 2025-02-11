from pathlib import Path
from glob import glob
import json
import torch
import asyncio
import time
import plotly.express as px


from nnsight import LanguageModel
from dataclasses import dataclass
from simple_parsing import field
from delphi.config import ExperimentConfig, LatentConfig, CacheConfig
from delphi.log.result_analysis import build_scores_df, feature_balanced_score_metrics
from delphi.__main__ import RunConfig, process_cache, populate_cache
from delphi.autoencoders.eleuther import load_and_hook_sparsify_models
from transformers import BitsAndBytesConfig, AutoTokenizer

from clearnets.train.sparse_gptneox import SparseGPTNeoXForCausalLM

from clearnets.autointerp.load_and_hook import load_and_hook_clearnet

@dataclass
class CustomModelRunConfig:
    tokenizer_model: str = field(
        default="EleutherAI/Meta-Llama-3-8B",
        positional=True,
    )


def load_artifacts(run_cfg: RunConfig, custom_run_cfg: CustomModelRunConfig):
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"


    tokenizer = AutoTokenizer.from_pretrained(custom_run_cfg.tokenizer_model)
    model = SparseGPTNeoXForCausalLM.from_pretrained(
        run_cfg.model,
        device_map={"": f"cuda"},
    )

    model = LanguageModel(
        model, 
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=run_cfg.hf_token,
        dispatch=True,
        tokenizer=tokenizer
    )

    if run_cfg.sparse_model == "":
        submodule_name_to_submodule, model = load_and_hook_clearnet(
            model,
            run_cfg.hookpoints,
        )
    else:
        submodule_name_to_submodule, model = load_and_hook_sparsify_models(
            model,
            run_cfg.sparse_model,
            hookpoints=run_cfg.hookpoints,
        )

    return run_cfg.hookpoints, submodule_name_to_submodule, model, model.tokenizer


async def test_clearnet():
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:1%]",
        dataset_row="text",
        batch_size=8,
        ctx_len=256,
        n_splits=5,
        n_tokens=10_000_000,
    )
    experiment_cfg = ExperimentConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=40,
        n_examples_test=50,
    )
    overwrite = []
    # overwrite.append("cache")
    # overwrite.append("scores")
    
    # for model, sparse_model in [
    #     (
    #         "/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/sparse-checkpoint-164000",
    #         "",
    #     ),
    #     ( 
    #         "/mnt/ssd-1/nora/dense-ckpts/checkpoint-118000",
    #         "/",
    #     ),


    run_cfg = RunConfig(
        name='clearnet-1',
        overwrite=overwrite,
        model="/mnt/ssd-1/caleb/clearnets/Dense-FineWebEduDedup-58M-s=42/sparse-checkpoint-164000",
        sparse_model="",
        explainer_model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        hookpoints=["layers.3.mlp"],
        explainer_model_max_len=4208,
        max_latents=100,
        seed=22,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True
    )
    custom_run_cfg = CustomModelRunConfig(
        tokenizer_model="EleutherAI/FineWeb-restricted",
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
    hookpoints, submodule_name_to_submodule, hooked_model, tokenizer = load_artifacts(
        run_cfg, custom_run_cfg
    )

    if (
        not glob(str(latents_path / ".*")) + glob(str(latents_path / "*"))
        or "cache" in run_cfg.overwrite
    ):
        populate_cache(
            run_cfg,
            cache_cfg,
            hooked_model,
            submodule_name_to_submodule,
            latents_path,
            tokenizer,
            filter_bos=run_cfg.filter_bos,
        )
    else:
        print(f"Files found in {latents_path}, skipping cache population...")

    latent_cfg = LatentConfig(
        width=list(submodule_name_to_submodule.values())[0].ae.width,
        min_examples=200,
        max_examples=10_000,
    )
    del hooked_model, submodule_name_to_submodule

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
            hookpoints,
            tokenizer,
            feature_range,
        )
    else:
        print(f"Files found in {scores_path}, skipping...")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    scores_path =  Path("results") / run_cfg.name / "scores"

    df = build_scores_df(scores_path, hookpoints, feature_range)
    

    for score_type in df["score_type"].unique():
        score_df = df[df['score_type'] == score_type]
        feature_balanced_score_metrics(score_df, score_type)

        fig = px.histogram(
            df[df["score_type"] == score_type],
            x="accuracy",
            barmode="overlay",
            title=f"Accuracy Distribution - {score_type}",
            nbins=100,
        )
        fig.write_image(visualize_path / f"autointerp_accuracies_{score_type}.pdf", format="pdf")



if __name__ == "__main__":
    asyncio.run(test_clearnet())