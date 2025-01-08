import os
import copy
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import torch
import numpy as np
import einops
import tqdm.auto as tqdm
import plotly.express as px
import plotly.express as px
import plotly.io as pio
from clearnets.transformers import CustomTransformer, TransformerConfig
import torch.nn.functional as F
import lovely_tensors as lt

from safetensors.torch import load_model
from sae import SaeConfig, SaeTrainer, TrainConfig
from datasets import Dataset as HfDataset

from clearnets.utils.utils import assert_type


lt.monkey_patch()

pio.templates['plotly'].layout.xaxis.title.font.size = 20 # type: ignore
pio.templates['plotly'].layout.yaxis.title.font.size = 20 # type: ignore
pio.templates['plotly'].layout.title.font.size = 30 # type: ignore

TRAIN_MODEL = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def parse_args():
    parser = ArgumentParser(description="Train the model")
    parser.add_argument("--model_seed", type=int, default=999, help="Seed for the model")
    parser.add_argument("--run_name", type=str, default='', help="Label for the run artifacts")
    return parser.parse_args()


def train_model():
    args = parse_args()
    run_identifier = f"{args.model_seed}{'-' + args.run_name if args.run_name else ''}"

    # Define the location to save the model, using a relative path
    PTH_LOCATION = f"workspace/grok/{run_identifier}.pth"
    os.makedirs(Path(PTH_LOCATION).parent, exist_ok=True)

    """# Model Training

    ## Config
    """
    p = 113
    frac_train = 0.3

    # Optimizer config
    lr = 1e-3
    # from the original paper; lower values delay the onset of grokking
    wd = 1.
    betas = (0.9, 0.95)

    num_epochs = 25000
    checkpoint_every = 100

    DATA_SEED = 598

    """## Define Task
    * Define modular addition
    * Define the dataset & labels

    Input format:
    |a|b|=|
    """

    a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
    b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
    equals_vector = einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)

    dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)

    labels = (dataset[:, 0] + dataset[:, 1]) % p

    """Convert this to a train + test set - 30% in the training set"""

    torch.manual_seed(DATA_SEED)
    indices = torch.randperm(p*p)
    cutoff = int(p*p*frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]
    test_data = dataset[test_indices]
    test_labels = labels[test_indices]

    """## Define Model"""
    # n_ctx
    # 5 yields drop around 13k
    # 4 yields fast drop around 8k
    # 3 yields fast drop around 5k
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.model_seed)
    import random; random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    config = TransformerConfig(
        d_vocab=p + 1,
        d_model=128,
        n_ctx=5, # dummy inputs hardcoded in huggingface transformers expect > 4
        num_layers=1,
        num_heads=4,
        d_mlp=512,
        act_type="ReLU",
        use_ln=False,
    )
    model = CustomTransformer(config)

    model.to(device)

    """Disable the biases, as we don't need them for this task and it makes things easier to interpret."""

    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    """## Define Optimizer + Loss"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    def loss_fn(logits, labels):
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        logits = logits.to(torch.float64)
        return F.cross_entropy(logits, labels).mean()
    train_logits = model(train_data).logits[..., :-1]
    # train_logits = model(train_data)
    train_loss = loss_fn(train_logits, train_labels)
    print(train_loss)
    test_logits = model(test_data).logits[..., :-1]
    # test_logits = model(test_data)
    test_loss = loss_fn(test_logits, test_labels)
    print(test_loss)

    print("Uniform loss:")
    print(np.log(p))

    """## Actually Train

    **Weird Decision:** Training the model with full batch training rather than stochastic gradient descent. We do this so to make training smoother and reduce the number of slingshots.
    """

    train_losses = []
    test_losses = []
    model_checkpoints = []
    checkpoint_epochs = []

    ablation_loss_increases = []
    if TRAIN_MODEL:
        for epoch in tqdm.tqdm(range(num_epochs)):
            # train_logits = model(train_data)
            train_logits = model(train_data).logits[..., :-1]
            train_loss = loss_fn(train_logits, train_labels)
            train_loss.backward()
            train_losses.append(train_loss.item())

            optimizer.step()
            optimizer.zero_grad()

            with torch.inference_mode():
                # test_logits = model(test_data)
                test_logits = model(test_data).logits[..., :-1]
                test_loss = loss_fn(test_logits, test_labels)
                test_losses.append(test_loss.item())

            if ((epoch + 1) % checkpoint_every) == 0:
                checkpoint_epochs.append(epoch)
                model_checkpoints.append(copy.deepcopy(model.state_dict()))
                print(f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}")

        torch.save(
            {
                "model": model.state_dict(),
                "dataset": dataset,
                "labels": labels,
                "config": config,
                "run_identifier": run_identifier,
                "checkpoints": model_checkpoints,
                "checkpoint_epochs": checkpoint_epochs,
                "test_losses": test_losses,
                "train_losses": train_losses,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "ablation_loss_increases": ablation_loss_increases
            },
            PTH_LOCATION)
    if not TRAIN_MODEL:
        cached_data = torch.load(PTH_LOCATION)
        model.load_state_dict(cached_data['model'])
        model_checkpoints = cached_data["checkpoints"]
        checkpoint_epochs = cached_data["checkpoint_epochs"]
        test_losses = cached_data['test_losses']
        train_losses = cached_data['train_losses']
        train_indices = cached_data["train_indices"]
        test_indices = cached_data["test_indices"]
        ablation_loss_increases = cached_data["ablation_loss_increases"]

    """## Show Model Training Statistics, Check that it groks!"""

    print(len(test_losses))
    print(len(train_losses))

    epochs = np.arange(0, len(train_losses), 100)
    df = pd.DataFrame({
        'Epoch': np.concatenate([epochs, epochs]),
        'Loss': np.array([train_losses[0 : len(train_losses) : 100] + test_losses[0 : len(test_losses) : 100]]).squeeze(),
        'Type': np.array(['Train'] * len(epochs) + ['Test'] * len(epochs))
        }
    )

    fig = px.line(
        df, 
        x='Epoch', y='Loss', color='Type', title='Training Curve for Modular Addition', 
        log_y=True
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend_title='',
        hovermode='x unified'
    )

    memorization_end_epoch = 1500
    circuit_formation_end_epoch = 12500
    cleanup_end_epoch = 18_500

    def add_lines(figure):
        figure.add_vline(memorization_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(circuit_formation_end_epoch, line_dash="dash", opacity=0.7)
        figure.add_vline(cleanup_end_epoch, line_dash="dash", opacity=0.7)
        return figure

    fig = add_lines(fig)
    fig.write_image(f"loss_curves-{run_identifier}.pdf", format="pdf")



def train_saes():
    def get_args():
        parser = ArgumentParser()
        parser.add_argument("--model_seed", type=int, default=1, help="Model seed to load checkpoints for")
        parser.add_argument("--finetune", action="store_true", help="Whether to finetune the model from the previous checkpoint\
                            or train from scratch") 
        parser.add_argument("--model_run_name", type=str, default='')
        parser.add_argument("--sae_run_name", type=str, default='')
        parser.add_argument("--sae_suffix", type=str, default='')
        return parser.parse_args()

    args = get_args()
    torch.manual_seed(args.model_seed)

    # Use existing on-disk model checkpoints
    model_identifier = f"{args.model_seed}{'-' + args.model_run_name if args.model_run_name else ''}"
    cached_data = torch.load(model_path := Path(f"workspace/grok/{model_identifier}.pth"))

    # Create SAE training dataset using multiple epochs over the training data
    num_epochs = 256 if args.finetune else 128
    train_indices = cached_data['train_indices']
    train_labels = cached_data['labels'][train_indices]
    train_data = cached_data['dataset'][train_indices]
    sae_train_dataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1),
        "labels": train_labels.repeat(num_epochs),
    })
    sae_train_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    # Load models from existing on-disk checkpoints
    config = cached_data['config']
    config = assert_type(TransformerConfig, config)
    model = CustomTransformer(config)

    model_checkpoints = cached_data["checkpoints"][::10]
    checkpoint_epochs = cached_data["checkpoint_epochs"][::10]   

    for i, (epoch, state_dict) in tqdm(enumerate(zip(checkpoint_epochs, model_checkpoints))):
        model.load_state_dict(state_dict)
        model.cuda() # type: ignore

        # Train SAE on checkpoints
        run_name = Path(f"sae/{model_identifier}/grok.{epoch}\
{'-' + args.sae_run_name if args.sae_run_name else ''}\
{'.finetune' if args.finetune else ''}\
{'.' + args.sae_suffix if args.sae_suffix else ''}")
        if run_name.exists():
            continue

        cfg = TrainConfig(
            SaeConfig(multi_topk=True),
            batch_size=64,
            run_name=str(run_name),
            log_to_wandb=True,
            hookpoints=[
                "blocks.*.hook_resid_pre", # start of block
                "blocks.*.hook_resid_mid", # after ln / attention
                "blocks.*.hook_resid_post", # after ln / mlp
                "blocks.*.hook_mlp_out"
            ],
        )
        trainer = SaeTrainer(cfg, sae_train_dataset, model)
            
        # finetune from previous checkpoints
        if args.finetune and i > 0:
            prev_run_name = Path(f"sae/{model_identifier}/grok.{checkpoint_epochs[i - 1]}")
            for name, sae in trainer.saes.items():
                load_model(sae, f"{prev_run_name}/{name}/sae.safetensors", device=str(device))

            trainer.cfg.lr = 1e-5
        
        trainer.fit()


if __name__ == '__main__':
    train_model()
    # train_saes()
