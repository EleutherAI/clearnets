import os
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from nnsight import LanguageModel
# from sae import SaeConfig, Sae, TrainConfig, SaeTrainer
from sae.trainer import SaeTrainer
from sae.sae import Sae
from sae.config import SaeConfig, TrainConfig
from datasets import Dataset as HfDataset
import lovely_tensors as lt

from clearnets.utils.utils import set_seeds
from clearnets.transformers import CustomTransformer

from clearnets.inference.inference import get_sae_metrics
from clearnets.plot.plot_modular_addition_grok import plot_modular_addition_grok

import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
lt.monkey_patch()
set_seeds(598)
device = torch.device("cuda")


def compute_losses(model, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch['input_ids'].to(device)

            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs, targets).logits.to(torch.float64)

            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def inference(args):
    run_identifier = f"{args.model_seed}{'-' + args.run_name if args.run_name else ''}"

    # torch.load hack to load model checkpoints from before module rename
    import sys; sys.modules['clearnets.grok'] = sys.modules['clearnets']
    # Use existing on-disk model checkpoints
    MODEL_PATH = Path(f"workspace/grok/{run_identifier}.pth")
    cached_data = torch.load(MODEL_PATH)
    
    OUT_PATH = Path(f"workspace/inference/{run_identifier}.pth")
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    model_checkpoints = cached_data["checkpoints"][::5]
    checkpoint_epochs = cached_data["checkpoint_epochs"][::5]

    debug_early_index = len(model_checkpoints)
    model_checkpoints = model_checkpoints[:debug_early_index]
    checkpoint_epochs = checkpoint_epochs[:debug_early_index]    
    
    checkpoint_data = torch.load(OUT_PATH) if OUT_PATH.exists() else {}

    config = cached_data['config']
    dataset = cached_data['dataset']
    labels = cached_data['labels']
    train_indices = cached_data['train_indices']
    test_indices = cached_data['test_indices']

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]
    print(len(train_data), "train data items")

    # SAEs get lower FVU on multiple epochs
    num_epochs = 128
    sae_train_dataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1),
        "labels": train_labels.repeat(num_epochs),
    })
    sae_train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    # Subset of data to collect eval metrics
    dataset = HfDataset.from_dict({
        "input_ids": train_data,
        "labels": train_labels,
    })
    dataset.set_format(type="torch", columns=["input_ids", "labels"])
    train_dl = DataLoader(dataset, batch_size=64, drop_last=True) # type: ignore

    test_data = dataset[test_indices]
    test_labels = labels[test_indices]
    test_dataset = HfDataset.from_dict({
        "input_ids": test_data,
        "labels": test_labels,
    })
    test_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    test_dl = DataLoader(test_dataset, batch_size=64, drop_last=True) # type: ignore

    model = CustomTransformer(config)

    dataloaders = {
        'train': train_dl,
        'test': test_dl
    }

    # NNSight requires a tokenizer. We are passing in tensors so any tokenizer will do
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))):
        if epoch not in checkpoint_data:
            checkpoint_data[epoch] = {}

        # Load model and SAEs
        model.load_state_dict(state_dict)
        model.to(device) # type: ignore
        nnsight_model = LanguageModel(model, tokenizer=tokenizer)

        sae_path = Path(f"sae/{run_identifier}{'_' + args.sae_name if args.sae_name else ''}/grok.{epoch}")
        if not sae_path.exists():
            cfg = TrainConfig(
                SaeConfig(multi_topk=True, expansion_factor=128, k=model.config.d_model), 
                batch_size=64,
                run_name=str(sae_path),
                log_to_wandb=True,
                hookpoints=[
                    "blocks.*.hook_resid_pre", # start of block
                    "blocks.*.hook_attn_out", # after ln / attention
                    "blocks.*.hook_mlp_out" # basically the same as resid_post but without the resid added
                ],
            )
            trainer = SaeTrainer(cfg, sae_train_dataset, model)
            trainer.fit()
        
        attns = [block.hook_attn_out for block in nnsight_model.blocks]
        mlps = [block.hook_mlp_out for block in nnsight_model.blocks]

        dictionaries = {}
        for i in range(len(nnsight_model.blocks)): # type: ignore
            dictionaries[attns[i]] = Sae.load_from_disk(
                os.path.join(sae_path, f'blocks.{i}.hook_attn_out'),
                device=device
            )
            dictionaries[mlps[i]] = Sae.load_from_disk(
                os.path.join(sae_path, f'blocks.{i}.hook_mlp_out'),
                device=device
            )
        all_submods = [submod for layer_submods in zip(attns, mlps) for submod in layer_submods]
        
        # Collect metrics
        checkpoint_data[epoch]['train_loss'] = compute_losses(model, train_dl, device)
        checkpoint_data[epoch]['test_loss'] = compute_losses(model, test_dl, device)
        metrics, activations = get_sae_metrics(
            model, 
            nnsight_model,
            dictionaries,
            all_submods,
            dataloaders,
        )
        checkpoint_data[epoch].update(metrics)

        torch.save(checkpoint_data, OUT_PATH)

    torch.save(checkpoint_data, OUT_PATH)

    
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_seed", type=int, default=1, help="Model seed to load checkpoints for")
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--sae_name", type=str, default='v2')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    inference(args)

    if args.plot:
        # Plots data over five model seeds
        plot_modular_addition_grok()
