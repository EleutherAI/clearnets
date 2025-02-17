from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from collections import defaultdict

import torch.nn.functional as F
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nnsight import LanguageModel
# from sparsify import Sae
from sparsify.sae import Sae
import lovely_tensors as lt
from sparsify.data import MemmapDataset
from torch.utils.data import DataLoader
import plotly.io as pio

from clearnets.utils.utils import set_seeds
from clearnets.language.hf_client import get_model_checkpoints
from clearnets.inference.inference import get_sae_metrics
from clearnets.plot.plot_pythia import plot_pythia
from clearnets.metrics import network_compression, singular_values


pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
lt.monkey_patch()
set_seeds(598)
device = torch.device("cuda")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def compute_losses(model, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch['input_ids'].to(device)

            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs, targets).logits

            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def load_pythia_dictionaries(nnsight_model, step_number: str, sae_path: Path, model_name: str = "pythia-160m"):
    dictionaries = {}
    layer_indices = list(range(len(nnsight_model.gpt_neox.layers)))[1::2] # type: ignore
    resids = [layer for layer in list(nnsight_model.gpt_neox.layers)[1::2]] # type: ignore

    for i, resid in zip(layer_indices, resids):
        dictionaries[resid] = Sae.load_from_disk(
            sae_path / f"{model_name} step {step_number} L{i}" / f'layers.{i}',
            device=device
        )

    return dictionaries, resids
    

@torch.no_grad()
def main():
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)
    model_name = "pythia-160m"
    sae_path = Path(f"/mnt/ssd-1/lucia/sae/")
    batch_size = 16  # Adjust based on your GPU memory

    args = get_args()

    OUT_PATH = Path(f"workspace/inference/{model_name}.pth")
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    checkpoints = get_model_checkpoints(f"EleutherAI/{model_name}")    
    log2_keys = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16_000, 33_000, 66_000, 131_000, 143_000]
    log_checkpoints = {key: checkpoints[key] for key in log2_keys}
    test_early_index = len(log2_keys)
    
    checkpoint_data = torch.load(OUT_PATH) if OUT_PATH.exists() else {}

    data = MemmapDataset("/mnt/ssd-1/pile_preshuffled/deduped/document.bin", ctx_len=2049)
    train_data = data.select(rng=range(100_000))
    test_data = data.select(rng=range(100_000, 100_000 + 2048))

    # Subset of data to collect eval metrics
    len_sample_data = 32 if args.debug else 512
    train_dl = DataLoader(train_data.select(range(len_sample_data)), batch_size=batch_size, drop_last=True) 
    test_dl = DataLoader(test_data.select(range(len_sample_data)), batch_size=batch_size, drop_last=True) 
    dataloaders = {
        "train": train_dl,
        "test": test_dl
    }

    load_dictionaries = partial(load_pythia_dictionaries, sae_path=sae_path)

    initial_model = AutoModelForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        torch_dtype=torch.float16,
        revision=f"step{log2_keys[0]}",
        cache_dir=".cache",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True) if "12b" in model_name else None
    )

    for step_number, checkpoint in tqdm(log_checkpoints.items()):
        if step_number not in checkpoint_data:
            checkpoint_data[step_number] = {}

        # Load model and SAEs
        model = AutoModelForCausalLM.from_pretrained(
            f"EleutherAI/{model_name}",
            torch_dtype=torch.float16,
            revision=checkpoint,
            cache_dir=".cache",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True) if "12b" in model_name else None
        )
        if not "12b" in model_name:
            model = model.cuda()

        # NNSight requires a tokenizer. We are passing in tensors so any tokenizer will do
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        nnsight_model = LanguageModel(model, tokenizer=tokenizer)

        dictionaries, all_submods = load_dictionaries(nnsight_model, step_number)

        if not args.debug:
            checkpoint_data[step_number]['train_loss'] = compute_losses(model, train_dl, device)
            checkpoint_data[step_number]['test_loss'] = compute_losses(model, test_dl, device)
            metrics, activations = get_sae_metrics(
                model, 
                nnsight_model,
                dictionaries,
                all_submods,
                dataloaders,
                feature_dims=[2],
                instance_dims=[0, 1],
            )
            checkpoint_data[step_number].update(metrics)

        layer_metrics = defaultdict(list)
        for i, submod in enumerate(model.gpt_neox.layers):
            parameters_initial = initial_model.gpt_neox.layers[i].attention.dense.weight.to(device)
            parameters_final = submod.attention.dense.weight

            layer_metrics["attn_layer_ranks"].append(network_compression(parameters_final, parameters_initial))
            layer_metrics["attn_max_rank"].append(min(parameters_final.size()))
            layer_metrics["attn_diff_singular_values"].append(singular_values(parameters_final - parameters_initial))
            layer_metrics["attn_singular_values"].append(singular_values(parameters_final))
            layer_metrics["attn_layer_norms"].append(torch.linalg.matrix_norm(parameters_final).item())

            parameters_initial = initial_model.gpt_neox.layers[i].mlp.dense_4h_to_h.weight.to(device).data
            parameters_final = submod.mlp.dense_4h_to_h.weight
            
            layer_metrics["mlp_layer_ranks"].append(network_compression(parameters_final, parameters_initial))
            layer_metrics["mlp_max_rank"].append(min(parameters_final.size()))
            layer_metrics["mlp_diff_singular_values"].append(singular_values(parameters_final - parameters_initial))
            layer_metrics["mlp_singular_values"].append(singular_values(parameters_final))
            layer_metrics["mlp_layer_norms"].append(torch.linalg.matrix_norm(parameters_final).item())
            
        checkpoint_data[step_number].update(layer_metrics)

        torch.save(checkpoint_data, OUT_PATH)
    torch.save(checkpoint_data, OUT_PATH)
    
    if args.plot:
        plot_pythia(test_early_index)
    
if __name__ == "__main__":
    main()