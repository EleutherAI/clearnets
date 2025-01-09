from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict

import torch    
from torch import Tensor
import torchvision as tv
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import (
    ConvNextV2Config, ConvNextV2ForImageClassification,
    Swinv2Config, Swinv2ForImageClassification,
    ViTConfig, ViTForImageClassification
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import lovely_tensors as lt
from datasets import Dataset as HfDataset
from sae.sae import Sae
from sae.config import SaeConfig, TrainConfig
from sae.trainer import SaeTrainer
# from sae import SaeConfig, SaeTrainer, TrainConfig
from nnsight import NNsight

from clearnets.utils import set_seeds
from clearnets.train.lightning_wrapper import ScheduleFreeLightningWrapper
from clearnets.generalization.metrics import network_compression, singular_values


torch.set_float32_matmul_precision('high')
lt.monkey_patch()


def vit_hookpoints(model):
    hookpoints = []
    submods = []
    for i, submod in enumerate(model.vit.encoder.layer):
        hookpoints.append(f"vit.encoder.layer.{i}.attention.output")
        hookpoints.append(f"vit.encoder.layer.{i}.output")

        submods.append(submod.attention.output)
        submods.append(submod.output)

    return hookpoints, submods

def swin_hookpoints(model):
    hookpoints = []
    submods = []
    for i, stage in enumerate(model.swinv2.encoder.layers):
        for j, submod in enumerate(stage.blocks):
            hookpoints.append(f"swinv2.encoder.layers.{i}.blocks.{j}.attention.output")
            hookpoints.append(f"swinv2.encoder.layers.{i}.blocks.{j}.output")

            submods.append(submod.attention.output)
            submods.append(submod.output)

    return hookpoints, submods

def convnext_hookpoints(model):
    hookpoints = []
    submods = []
    for i, stage in enumerate(model.convnextv2.encoder.stages):
        for j, submod in enumerate(stage.layers):
            hookpoints.append(f"convnextv2.encoder.stages.{i}.layers.{j}.pwconv2")

            submods.append(submod.pwconv2)

    return hookpoints, submods


def load_dictionaries(model_name, model, sae_path, device) -> tuple[dict, list]:
    dictionaries = {}
    hookpoints, submods = {
        'ViT': vit_hookpoints,
        'Swin': swin_hookpoints,
        'ConvNeXtV2': convnext_hookpoints
    }[model_name](model)

    for hookpoint, submod in zip(hookpoints, submods):
        dictionaries[submod] = Sae.load_from_disk(sae_path / hookpoint, device)

    return dictionaries, submods


def create_matched_models(image_size=32, num_labels=10):
    """
    Creates ConvNeXtV2, Swin, and ViT models with roughly matched parameter counts
    using Hugging Face Transformers library.
    Returns models and their parameter counts.
    """
    # ConvNeXtV2-Tiny configuration (~28M parameters)
    convnext_config = ConvNextV2Config(
        num_channels=3,
        num_labels=num_labels,
        depths=[3, 3, 9, 3],
        hidden_sizes=[96, 192, 384, 768],
        layer_norm_eps=1e-6,
        image_size=image_size
    )
    convnext_model = ConvNextV2ForImageClassification(convnext_config)
    
    # Swin-Tiny configuration (~28M parameters)
    swin_config = Swinv2Config(
        image_size=image_size,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        num_labels=num_labels
    )
    swin_model = Swinv2ForImageClassification(swin_config)
    
    # ~27M parameters
    vit_config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        num_channels=3,
        hidden_size=432,  # Increased from 384
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=1728,  # 4 * hidden_size
        num_labels=num_labels
    )
    vit_model = ViTForImageClassification(vit_config)

    models = {
        # 'ConvNeXtV2': convnext_model,
        'Swin': swin_model,
        'ViT': vit_model
    }
    torch.save(models, Path('data') / 'vision_cifar10' / 'model_initializations.pt')

    param_counts = {
        name: sum(p.numel() for p in model.parameters())
        for name, model in models.items()
    }

    hookpoints = {
        'ViT': vit_hookpoints(vit_model)[0],
        'Swin': swin_hookpoints(swin_model)[0],
        'ConvNeXtV2': convnext_hookpoints(convnext_model)[0]
    }
    return {
        'models': models,
        'param_counts': param_counts,
        'configs': {
            'ConvNeXtV2': convnext_config,
            'Swin': swin_config,
            'ViT': vit_config
        },
        'hookpoints': hookpoints,
        'feature_dims': {
            'ViT': [2],
            'Swin': [1, 2],
            # Treat dimension 1-2 as an instance dimension to get local spatial information over an image patch, 
            # and as a feature dimension to get combined features for each individual images
            'ConvNeXtV2': [3]
        },
    }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sae", action="store_true")
    parser.add_argument("--seed", type=int, default=38)

    return parser.parse_args()


def train_vision(args, metadata, train, val, test):
    learning_rate = 1e-5
    betas = (0.95, 0.999)
    warmup_steps = 10_000

    # Convert train to a HF Dataset
    sae_train = HfDataset.from_dict({
        "input_ids": torch.stack([x for x, _ in train]),
        "label": torch.tensor([y for _, y in train]),
    })
    sae_train.set_format(type="torch", columns=["input_ids", "label"])

    dm: pl.LightningDataModule = pl.LightningDataModule.from_datasets(
        train,
        val,
        test,
        batch_size=128, 
        num_workers=47,
    )

    

    for name, model in metadata['models'].items():
        # seed=38 models trained in 32bit, didn't affect perf
        lightning_model = ScheduleFreeLightningWrapper(model, learning_rate, betas, warmup_steps)
        wandb_name = f"{name}_s={args.seed}"
        ckpts_path = Path('arch-evals') / wandb_name
        if not ckpts_path.exists() or args.overwrite:
            trainer = pl.Trainer(
                devices=[0, ],
                logger=WandbLogger(
                    name=wandb_name, project="arch-evals", entity="eleutherai", reinit=True
                ) if not args.debug else None,
                max_epochs=300,
                precision='bf16-mixed',
                deterministic=True,
                gradient_clip_val=None,
                callbacks=[
                    ModelCheckpoint(dirpath=ckpts_path, save_last=True, every_n_epochs=1, save_top_k=-1), 
                    EarlyStopping(monitor='val_loss', patience=30)
                ],
            )

            trainer.fit(lightning_model, dm)
            trainer.test(lightning_model, dm)

            try:
                wandb.finish()
            except Exception as e:
                print(e)


        if args.sae:
            assert metadata['hookpoints'][name] != [], "Hookpoints not set for model"
            cfg = TrainConfig(
                SaeConfig(multi_topk=True, num_latents=4096),
                batch_size=8,
                run_name=str(Path('sae') / wandb_name),
                log_to_wandb=not args.debug,
                hookpoints=metadata['hookpoints'][name],
                grad_acc_steps=2,
                micro_acc_steps=2,
            )
            trainer = SaeTrainer(
                cfg, sae_train, model.cuda(), 
                dummy_inputs={'pixel_values': torch.randn(1, 3, 32, 32)}
            )

            trainer.fit()


@torch.no_grad()
def inference_vision(args, metadata: dict, train, test):
    device = torch.device("cuda")
    # Fucking with data formats because I'm lazy
    len_dataset = 4096
    sae_train = HfDataset.from_dict({
        "input_ids": torch.stack([x for x, _ in train]),
        "label": torch.tensor([y for _, y in train]),
    })
    sae_train.set_format(type="torch", columns=["input_ids", "label"])

    # Just for evaluating loss bc :) 
    dm: pl.LightningDataModule = pl.LightningDataModule.from_datasets(
        train_dataset=train,
        test_dataset=test,
        batch_size=128, 
        num_workers=47,
    )

    train = HfDataset.from_dict({
        "input_ids": torch.stack([x for x, _ in train])[:len_dataset],
        "label": torch.tensor([y for _, y in train])[:len_dataset],
    })
    train.set_format(type="torch", columns=["input_ids", "label"])
    test = HfDataset.from_dict({
        "input_ids": torch.stack([x for x, _ in test])[:len_dataset],
        "label": torch.tensor([y for _, y in test])[:len_dataset],
    })
    test.set_format(type="torch", columns=["input_ids", "label"])

    train_dl = torch.utils.data.DataLoader(train, batch_size=128) # type: ignore
    test_dl = torch.utils.data.DataLoader(test, batch_size=128) # type: ignore
    
    inference_data = {}
    # if (Path('data') / 'vision_cifar10' / 'activation_data.pt').exists():
    activation_data = torch.load(Path('data') / 'vision_cifar10' / 'activation_data.pt')
    model_initialization_data = torch.load(Path('data') / 'vision_cifar10' / 'model_initializations.pt')

    for name, model_cls in metadata['models'].items():
        ckpts_path = Path('arch-evals') / f"{name}_s={args.seed}"

        model = ScheduleFreeLightningWrapper.load_from_checkpoint(
            str(ckpts_path / "last.ckpt"),
            model=model_cls.model,
            learning_rate=model_cls.learning_rate,
            betas=model_cls.betas,
            warmup_steps=model_cls.warmup_steps,
            map_location=device
        )
        model.cuda()

        trainer = pl.Trainer(
            accelerator='auto',
            devices=1,
            enable_checkpointing=False,
            logger=False
        )
        
        nnsight_model = NNsight(model)
        
        dictionaries, submods = load_dictionaries(
            name, 
            nnsight_model.model, 
            Path('sae') / f"{name}_s={args.seed}", 
            device,
        )
        
        test_loss = trainer.test(model, dataloaders=dm.test_dataloader())[0]['test_loss']
        train_loss = trainer.test(model, dataloaders=dm.train_dataloader())[0]['test_loss']

        print(f"{name} train loss: {train_loss:.4f}")
        print(f"{name} test loss: {test_loss:.4f}")


        nnsight_model.cuda()
        dataloaders = {
            'train': train_dl,
            'test': test_dl
        }
        metrics, activations = {}, {}
        # metrics, activations = get_sae_metrics(
        #     model.model, 
        #     nnsight_model,
        #     dictionaries,
        #     submods,
        #     dataloaders,
        #     feature_dims=feature_dims,
        #     device=device
        # )

        metrics['num_sae_features'] = sum(d.num_latents for d in dictionaries.values())
        metrics['sae_k'] = sum(d.cfg.k for d in dictionaries.values())

        # In vision models these often look like [feature map batch, feature map channel, feature map height, feature map width]
        # And don't match the input size (feature map batch is the number of convolutions when you slide over the image * original batch and it
        # changes over the layers)
        # We can probably flatten the height and width since they're only not flattened to maintain the spatial bias

        # To get their rank we need to flatten some and batch other dimensions. In this case we flatten the feature map channel
        # and batch the feature map height and width

        def get_layer_metrics(parameters: Tensor, parameters_initial: Tensor, prefix: str):
            return {
                f"{prefix}_layer_ranks": network_compression(parameters, parameters_initial),
                f"{prefix}_max_rank": min(parameters.size()),
                f"{prefix}_diff_singular_values": singular_values(parameters - parameters_initial),
                f"{prefix}_singular_values": singular_values(parameters),
                f"{prefix}_layer_norms": torch.linalg.matrix_norm(parameters).item()
            }

        layer_metrics = defaultdict(list)
        if name == 'ConvNeXtV2':
            # Parameters with shape [m, 1, n n] are channels and kernels and the kernels need to be flattened
            # Parameters with two dims are left as is
            # Bias terms with one dim are discarded
            # Parameters with shape [m, n, o, o] are like the first option but with spatial downsampling. o and o can clearly be flattened buttt
            for i, stage in enumerate(model.model.convnextv2.encoder.stages):
                for j, submod in enumerate(stage.layers):
                    parameters_initial = model_initialization_data[
                        name
                    ].model.convnextv2.encoder.stages[i].layers[j].pwconv2.weight.to(device)
                    parameters_final = submod.pwconv2.weight
                    layer_metrics.update(
                        get_layer_metrics(parameters_final, parameters_initial, 'conv')
                    )
        elif name == 'Swin':
            for i, stage in enumerate(model.model.swinv2.encoder.layers):
                for j, submod in enumerate(stage.blocks):
                    parameters_initial = model_initialization_data[name].model.swinv2.encoder.layers[i].blocks[j].attention.output.dense.weight.to(device)
                    parameters_final = submod.attention.output.dense.weight
                    layer_metrics.update(
                        get_layer_metrics(parameters_final, parameters_initial, 'attn')
                    )

                    parameters_initial = model_initialization_data[name].model.swinv2.encoder.layers[i].blocks[j].output.dense.weight.to(device)
                    parameters_final = submod.output.dense.weight
                    layer_metrics.update(
                        get_layer_metrics(parameters_final, parameters_initial, 'mlp')
                    )

        elif name == 'ViT':
            for i, submod in enumerate(model.model.vit.encoder.layer):
                parameters_initial = model_initialization_data[name].model.vit.encoder.layer[i].attention.output.dense.weight.to(device)
                parameters_final = submod.attention.output.dense.weight
                layer_metrics.update(
                    get_layer_metrics(parameters_final, parameters_initial, 'attn')
                )

                parameters_initial = model_initialization_data[name].model.vit.encoder.layer[i].output.dense.weight.to(device)
                parameters_final = submod.output.dense.weight
                layer_metrics.update(
                    get_layer_metrics(parameters_final, parameters_initial, 'mlp')
                )
        else:
            raise ValueError(f"Unknown model name: {name}")

        metrics.update(layer_metrics)

        inference_data[name] = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            **metrics
        }
        activation_data[name] = activations
    
    torch.save(inference_data, Path('data') / 'vision_cifar10' / 'inference_data.pt')
    torch.save(activation_data, Path('data') / 'vision_cifar10' / 'activation_data.pt')


def main():
    args = parse_args()

    # 598, then 42, then incorporated into run name
    set_seeds(args.seed)

    image_size = 32
    metadata = create_matched_models(image_size)
    
    print("\nParameter counts:")
    for model_name, count in metadata['param_counts'].items():
        print(f"{model_name}: {count:,} parameters")
        
    # Load CIFAR-10 dataset
    nontest = CIFAR10(
        "/home/lucia/cifar10", download=True, transform=tv.transforms.ToTensor()
    )
    train, val = random_split(nontest, [0.9, 0.1])
    test = CIFAR10(
        "/home/lucia/cifar10-test", download=True, transform=tv.transforms.ToTensor()
    )

    train_vision(args, metadata, train, val, test)
    inference_vision(args, metadata, train, test)

if __name__ == "__main__":
    main()