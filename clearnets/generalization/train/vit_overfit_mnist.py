from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTConfig
from datasets import Dataset as HfDataset, load_dataset
from sae import SaeConfig, SaeTrainer, TrainConfig
from tqdm import tqdm
import lovely_tensors as lt

from clearnets.utils import set_seeds, assert_type


# Forgot to save this in pth
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



def create_balanced_sample(dataset: HfDataset, n):
    dataset.set_format(type='numpy', columns=['input_ids', 'label'])
    labels = dataset['label']
    unique_labels, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)
    samples_per_class = n // num_classes

    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    balanced_indices = []
    for label in unique_labels:
        if counts[label] >= samples_per_class:
            balanced_indices.extend(np.random.choice(class_indices[label], size=samples_per_class, replace=False))
        else:
            balanced_indices.extend(class_indices[label])
            balanced_indices.extend(np.random.choice(class_indices[label], 
                                                     size=samples_per_class - counts[label], 
                                                     replace=True))

    balanced_dataset = dataset.select(balanced_indices)
    balanced_dataset.set_format(type='torch', columns=['input_ids', 'label'])
    return balanced_dataset


def parse_args():
    parser = ArgumentParser(description="Train Vision Transformer on MNIST")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_name", type=str, default='')
    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lt.monkey_patch()
args = parse_args()
set_seeds(args.seed)

# Define hyperparameters
batch_size = 16
num_epochs = 100
learning_rate = 1e-4
checkpoint_interval = 5

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

test_dataset: HfDataset = load_dataset('mnist', split='test') # type: ignore

test_dataset = test_dataset.map(
    function=map_fn,
    remove_columns=['image'],
    new_fingerprint='transformed_mnist', # type: ignore
    keep_in_memory=True # type: ignore
)
test_dataset.set_format(type='torch', columns=['input_ids', 'label'])

# Calculate mean and std of pixel values
input_ids = assert_type(Tensor, train_dataset['input_ids'])
mean = input_ids.mean().item()
std = input_ids.std().item()
def normalize(image):
    transform = transforms.Compose([
        transforms.Normalize((mean,), (std,))
    ])
    return transform(image)

# Induce overfitting by training on a small subset of the train set
train_dataset = create_balanced_sample(train_dataset, 40)
train_dataset = train_dataset.map(
    lambda example: {'input_ids': normalize(example['input_ids'])},
    new_fingerprint='transformed_mnist'
)

test_dataset = test_dataset.map(
    lambda example: {'input_ids': normalize(example['input_ids'])},
    new_fingerprint='transformed_mnist'
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # type: ignore
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) # type: ignore

config = ViTConfig(
    image_size=28,
    patch_size=4,
    num_channels=1,
    num_labels=10,
    num_hidden_layers=4,
    hidden_size=512,
    num_attention_heads=4,
    intermediate_size=4096,
)
model = ViTForImageClassification(config).to(device)

criterion = nn.CrossEntropyLoss()
# Induce overfitting by zeroing weight decay
wd = 0.
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)

run_identifier = f"mnist_seed_{args.seed}{'_' + args.run_name if args.run_name else ''}"
checkpoints_path = Path("data/vit_ckpts") / run_identifier
checkpoints_path.mkdir(exist_ok=True)

checkpoints = []
checkpoint_epochs = []
test_losses = []
train_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        data, target = batch['input_ids'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data).logits
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += target.size(0)
        train_correct += predicted.eq(target).sum().item()

    train_accuracy = 100.0 * train_correct / train_total
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Save checkpoint every 5 epochs
    if (epoch + 1) % checkpoint_interval == 0:
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                data, target = batch['input_ids'].to(device), batch['label'].to(device)
                outputs = model(data).logits
                loss = criterion(outputs, target)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        test_accuracy = 100.0 * test_correct / test_total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        checkpoints.append(model.state_dict())
        checkpoint_epochs.append(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        checkpoint_path = checkpoints_path / f"mnist_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'test_loss': test_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        # Train an SAE
        run_name = Path(f"sae/vit/{run_identifier}/{epoch}")
        if run_name.exists():
            continue
            
        cfg = TrainConfig(
            SaeConfig(multi_topk=True),
            batch_size=16,
            run_name=str(run_name),
            log_to_wandb=False,
            layers=[0, 1, 2, 3]
        )

        dummy_inputs={'pixel_values': torch.randn(3, 1, 28, 28)}

        trainer = SaeTrainer(cfg, train_dataset, model, dummy_inputs)
        trainer.fit()

print("Training completed!")

torch.save(
{
    "model": model.state_dict(),
    "dataset": train_dataset,
    "config": config,
    "run_identifier": run_identifier,
    "checkpoints": checkpoints,
    "checkpoint_epochs": checkpoint_epochs,
    "test_losses": test_losses,
    "train_losses": train_losses,
},
checkpoints_path / "final.pth")