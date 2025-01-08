from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from datasets import Dataset as HfDataset
from torch.utils.data import DataLoader
from transformers import (
    ConvNextV2ForImageClassification, ConvNextV2Config, Swinv2Config, Swinv2ForImageClassification
)
import lovely_tensors as lt

from clearnets.pgdl.complexity_measures import get_generalization_score
from clearnets.train.lightning_wrapper import ScheduleFreeLightningWrapper

lt.monkey_patch()

# Load test dataset
test_dataset = datasets.CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)
train_dataset = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)


# Convert to HF Dataset
test_dataset = HfDataset.from_dict({
    "input_ids": torch.stack([sample[0] for sample in test_dataset]),
    "label": torch.tensor([sample[1] for sample in test_dataset])
})
test_dataset.set_format(type="torch", columns=["input_ids", "label"])

train_dataset = HfDataset.from_dict({
    "input_ids": torch.stack([sample[0] for sample in train_dataset]),
    "label": torch.tensor([sample[1] for sample in train_dataset])
})
train_dataset.set_format(type="torch", columns=["input_ids", "label"])

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load test model
image_size = 32
num_labels = 10
convnext_config = ConvNextV2Config(
    num_channels=3,
    num_labels=num_labels,
    depths=[3, 3, 9, 3],
    hidden_sizes=[96, 192, 384, 768],
    layer_norm_eps=1e-6,
    image_size=image_size
)
conv_model = ConvNextV2ForImageClassification(convnext_config)

lightning_wrapper = ScheduleFreeLightningWrapper(conv_model)
    
ckpts_path = Path('arch-evals') / 'ConvNeXtV2_s=38'
wrapper = ScheduleFreeLightningWrapper.load_from_checkpoint(
    str(ckpts_path / "last.ckpt"),
    model=conv_model,
    map_location="cuda"
)

conv_model = wrapper.model
conv_model.eval()

# Original code multiplied batch by 3 then downsampled the vectors for some reason lmao (only in complexity DB)
# I plan to split the batch into 3 and downsample 
print("ConvNeXtV2 train:", get_generalization_score(conv_model, train_dataloader))
print("ConvNeXtV2 test:", get_generalization_score(conv_model, test_dataloader))


# Load test model
image_size = 32
num_labels = 10

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

lightning_wrapper = ScheduleFreeLightningWrapper(swin_model)
    
ckpts_path = Path('arch-evals') / 'Swin_s=38'
wrapper = ScheduleFreeLightningWrapper.load_from_checkpoint(
    str(ckpts_path / "last.ckpt"),
    model=swin_model,
    map_location="cuda"
)

swin_model = wrapper.model
swin_model.eval()

# Original code multiplied batch by 3 then downsampled the vectors for some reason lmao (only in complexity DB)
# I plan to split the batch into 3 and downsample 
batch_size = num_labels * 10

print("Swin train:", get_generalization_score(swin_model, train_dataloader))
print("Swin test:", get_generalization_score(swin_model, test_dataloader))


