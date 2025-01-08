from argparse import ArgumentParser
from pathlib import Path
from functools import partial

import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from nnsight import NNsight
import torch
from torch import Tensor
from transformers import AutoTokenizer, ViTConfig, ViTForImageClassification
from torchvision import transforms
from sae.sae import Sae
import lovely_tensors as lt
from torch.utils.data import DataLoader
from datasets import Dataset as HfDataset
from sae.config import SaeConfig, TrainConfig
from sae.trainer import SaeTrainer

from clearnets.inference.inference import get_sae_metrics
from clearnets.plot.plot_mnist_vit import plot_mnist_vit
from clearnets.utils.utils import set_seeds, assert_type


lt.monkey_patch()
set_seeds(598)
device = torch.device("cuda")

def compute_losses(model, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(inputs).logits
            logits = logits.reshape(-1, logits.size(-1))
            
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def load_mnist_vit_dictionaries(epoch_model, epoch: str, sae_train_dataset: HfDataset, sae_path: Path):
    attn_hookpoints = [f"vit.encoder.layer.{i}.attention.output" for i in range(len(epoch_model.vit.encoder.layer))] # type: ignore
    mlp_hookpoints = [f"vit.encoder.layer.{i}.output" for i in range(len(epoch_model.vit.encoder.layer))] # type: ignore

    epoch_sae_path = sae_path / f'{epoch}'
    if not epoch_sae_path.exists():
        cfg = TrainConfig(
            SaeConfig(multi_topk=True), 
            batch_size=32,
            run_name=str(epoch_sae_path),
            log_to_wandb=not args.debug,
            hookpoints=attn_hookpoints + mlp_hookpoints
        )
        dummy_inputs={'pixel_values': torch.randn(3, 1, 28, 28)}
        trainer = SaeTrainer(cfg, sae_train_dataset, epoch_model, dummy_inputs)
        trainer.fit()
    
    attns = [block.attention.output for block in epoch_model.vit.encoder.layer]
    mlps = [block.output for block in epoch_model.vit.encoder.layer]
    all_submods = [submod for layer_submods in zip(attns, mlps) for submod in layer_submods] # [embed] + resids

    dictionaries = {}
    for i in range(len(epoch_model.vit.encoder.layer)): # type: ignore
        dictionaries[attns[i]] = Sae.load_from_disk(
            epoch_sae_path / f'vit.encoder.layer.{i}.attention.output',
            device
        )

        dictionaries[mlps[i]] = Sae.load_from_disk(
            epoch_sae_path / f'vit.encoder.layer.{i}.output',
            device
        )

    return all_submods, dictionaries


def get_test_dataset():
    train_dataset: HfDataset = load_dataset("mnist", split='train') # type: ignore

    def map_fn(ex):
        return {
            'input_ids': transforms.ToTensor()(ex['image']),
            'label': ex['label']
        }

    train_dataset = train_dataset.map(
        function=map_fn,
        remove_columns=['image'],
        new_fingerprint='transformed_mnist', # type: ignore
        keep_in_memory=True # type: ignore
    )
    train_dataset = train_dataset.with_format('torch')
    train_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    print("Final columns:", train_dataset.column_names)

    # Calculate mean and std of pixel values
    input_ids = assert_type(Tensor, train_dataset['input_ids'])
    mean = input_ids.mean().item()
    std = input_ids.std().item()
    def normalize(image):
        transform = transforms.Compose([
            transforms.Normalize((mean,), (std,))
        ])
        return transform(image)


    test_dataset: HfDataset = load_dataset('mnist', split='test') # type: ignore

    test_dataset = test_dataset.map(
        function=map_fn,
        remove_columns=['image'],
        new_fingerprint='transformed_mnist', # type: ignore
        keep_in_memory=True # type: ignore
    )
    test_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    test_dataset = test_dataset.map(
        lambda example: {'input_ids': normalize(example['input_ids'])},
        new_fingerprint='transformed_mnist'
    )

    return test_dataset


def inference(model_path: Path, out_path: Path, sae_path: Path):
    cached_data = torch.load(model_path)

    # Dataset 
    batch_size = 40 # Dataset is only of len 40 to cause overfitting
    dataset = cached_data['dataset']
    train_data = dataset['input_ids']
    # TODO increase data size
    print(len(train_data))
    train_labels = dataset['label']
    train_dl = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    test_dataset= get_test_dataset()
    test_dl = DataLoader(test_dataset, batch_size=batch_size, drop_last=True) # type: ignore
    
    # SAEs dataset
    num_epochs = 128
    sae_train_dataset: HfDataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1, 1, 1), # .flatten(1, 3)
        "label": train_labels.repeat(num_epochs),
    })
    sae_train_dataset.set_format(type="torch", columns=["input_ids", "label"])

    # Model
    config = cached_data['config']
    config = assert_type(ViTConfig, config)
    model = ViTForImageClassification(config)
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data['checkpoint_epochs']

    # NNSight requires a tokenizer, we are passing in tensors so any tokenizer will do
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    load_dictionaries = partial(load_mnist_vit_dictionaries, sae_train_dataset=sae_train_dataset, sae_path=sae_path)

    checkpoint_data = torch.load(out_path) if out_path.exists() else {}

    dataloaders = {
        'train': train_dl,
        'test': test_dl
    }
    
    for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))):
        model.load_state_dict(state_dict)
        model.cuda() # type: ignore
        nnsight_model = NNsight(model)

        all_submods, dictionaries = load_dictionaries(nnsight_model, checkpoint_epochs[0])
        
        if epoch not in checkpoint_data:
            checkpoint_data[epoch] = {}

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

        torch.save(checkpoint_data, out_path)

    torch.save(checkpoint_data, out_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_seed", type=int, default=1, help="Model seed to load checkpoints for")
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--sae_name", type=str, default='v3')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_identifier = f'mnist_seed_{args.model_seed}'

    # Use existing on-disk model checkpoints
    model_path = Path("data") / "vit_ckpts" / run_identifier / "final.pth"
    sae_path = Path("sae") / "vit" / f"{run_identifier}_{args.sae_name}"
    out_path = Path(f"workspace") / 'inference' / run_identifier / "inference.pth"
    images_path = Path("images")
    
    out_path.parent.mkdir(exist_ok=True, parents=True)
    images_path.mkdir(exist_ok=True)

    inference(model_path, out_path, sae_path)

    if args.plot:
        plot_mnist_vit(model_path, out_path, images_path)