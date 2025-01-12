from argparse import ArgumentParser
from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, load_from_disk
import pytorch_lightning as pl
import torchmetrics as tm
from schedulefree import AdamWScheduleFree
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
import lovely_tensors as lt
from sae.data import chunk_and_tokenize

from clearnets.train.sparse_gptneox import SparseGPTNeoForCausalLM
from clearnets.train.sparse_gptneox_config import SparseGPTNeoConfig
from clearnets.utils import set_seeds, assert_type

lt.monkey_patch()
torch.set_float32_matmul_precision("high")
SEED = 42
set_seeds(SEED)

class ThresholdCheckpoint(Callback):
    def __init__(self, threshold: float, dirpath: str):
        super().__init__()
        self.threshold = threshold
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
    def on_validation_end(self, trainer, pl_module):
        current_loss = trainer.callback_metrics.get('val_loss')
        
        if current_loss is not None and current_loss <= self.threshold:
            checkpoint_path = self.dirpath / f"val_loss_{current_loss:.4f}.ckpt"
            
            if not checkpoint_path.exists():
                trainer.save_checkpoint(str(checkpoint_path))
                print(f"\nValidation loss {current_loss:.4f} hit threshold {self.threshold:.4f}. Saved checkpoint to {checkpoint_path}")

tiny_stories_8m_config = {
    "_name_or_path": "//amlta41566503acb2986203fbd2fc58f9ff6/projects/CODE_YUANZHI/amlt-results/7318563093.69241-46ef7114-0cc8-4d54-8d19-c1863a28eb04/trainer_textbook/checkpoint-25750/",
    "activation_function": "gelu_new",
    "architectures": ["GPTNeoForCausalLM"],
    "attention_dropout": 0,
    "attention_layers": [
        "global",
        "local",
        "global",
        "local",
        "global",
        "local",
        "global",
        "local",
    ],
    "attention_types": [[["global", "local"], 4]],
    "bos_token_id": 50256,
    "embed_dropout": 0,
    "eos_token_id": 50256,
    "gradient_checkpointing": False,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": None,
    "layer_norm_epsilon": 1e-05,
    "max_position_embeddings": 2048,
    "model_type": "gpt_neo",
    "num_heads": 16,
    "num_layers": 8,
    "resid_dropout": 0,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "torch_dtype": "bfloat16",  # "float32",
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 50257,
    "window_size": 256,
}

 
def get_dataloaders(dataset_str: str, tokenizer, batch_size: int, ctx_len: int, num_workers=16):
    dataset: DatasetDict = load_dataset(dataset_str) # type: ignore

    cache_file_name=Path(f"data/{dataset_str.replace('/', '--')}/dataset.cache")
    cache_file_name.parent.mkdir(parents=True, exist_ok=True)

    if not cache_file_name.exists():
        processed_dataset = chunk_and_tokenize(
            dataset,
            tokenizer,
            max_seq_len=ctx_len,
            num_proc=num_workers,
            text_key="text" if "text" in dataset["train"].column_names else "story",
        )
        processed_dataset.save_to_disk(cache_file_name)
    else:
        processed_dataset = load_from_disk(cache_file_name)
    
    processed_dataset = assert_type(DatasetDict, processed_dataset)
    processed_dataset.set_format(type="torch", columns=["input_ids"])

    dataloaders = {}
    for split in processed_dataset.keys():
        dataloaders[split] = DataLoader(
            processed_dataset[split],
            batch_size=batch_size, 
            shuffle=split == "train", 
            num_workers=num_workers
        )
    
    return dataloaders['train'], dataloaders['validation'] if 'validation' in dataloaders else dataloaders['test']


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, tokenizer, lr: float = 5e-4, betas: tuple = (0.9, 0.95)):
        super().__init__()
        # From https://huggingface.co/roneneldan/TinyStories-33M
        # lr_scheduler = "constant"
        self.learning_rate = lr # 5e-4 in original
        self.weight_decay = 0.1
        self.betas = betas
        self.tokenizer = tokenizer
        self.model = model

        self.train_acc = tm.Accuracy(
            "multiclass", num_classes=tiny_stories_8m_config["vocab_size"]
        )
        self.val_acc = tm.Accuracy(
            "multiclass", num_classes=tiny_stories_8m_config["vocab_size"]
        )

    def forward(self, input_ids):
        # attention_mask=attention_mask, 
        return self.model(
            input_ids=input_ids, labels=input_ids
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
            logger=True,
        )

        self.log(
            "train_perplexity",
            torch.exp(outputs.loss),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
            logger=True,
        )

        logits = outputs.logits[:, :-1, :]
        self.train_acc(
            logits.reshape(-1, logits.size(-1)), 
            batch["input_ids"][:, 1:].reshape(-1)
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_epoch=True,
            on_step=False,
            logger=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
        )

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        self.log(
            "val_loss",
            outputs.loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
            logger=True,
        )
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
            logger=True,
        )
        self.log(
            "val_perplexity",
            torch.exp(outputs.loss),
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
            logger=True,
        )

    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            device = next(self.parameters()).device
            sample_input = torch.tensor([[self.tokenizer.bos_token_id]], device=device)
            # sample_output = self.model.generate(sample_input, max_new_tokens=99)
            sample_output = self.model.generate(
                sample_input,
                # attention_mask=torch.ones_like(sample_input),
                max_new_tokens=99,
            )
            sample_str = self.tokenizer.decode(
                sample_output[0], skip_special_tokens=True
            )
            print(f"\nEpoch {self.current_epoch} sample:", sample_str)

    def configure_optimizers(self):
        self.optimizer = AdamWScheduleFree(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            warmup_steps=2000,
        )
        return {
            "optimizer": self.optimizer,
        }

    def train(self, mode: bool = True) -> None:
        """Set the model to training mode"""
        self.model.train(mode)
        optimizer = self.optimizers()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        for opt in optimizer:
            opt.train(mode)

    def eval(self) -> None:
        """Set the model to evaluation mode"""
        self.model.eval()
        optimizer = self.optimizers()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        for opt in optimizer:
            opt.eval()

    def on_validation_model_eval(self) -> None:
        self.model.eval()
        optimizer = self.optimizers()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        for opt in optimizer:
            opt.eval()

    def on_validation_model_train(self) -> None:
        self.model.train()
        optimizer = self.optimizers()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        for opt in optimizer:
            opt.train()

    def on_test_model_eval(self) -> None:
        self.model.eval()
        optimizer = self.optimizers()

        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        for opt in optimizer:
            opt.eval()

    def on_test_model_train(self) -> None:
        self.model.train()
        optimizer = self.optimizers()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        for opt in optimizer:
            opt.train()

    def on_predict_model_eval(self) -> None:  # redundant with on_predict_start()
        self.model.eval()
        optimizer = self.optimizers()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        for opt in optimizer:
            opt.eval()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--config", type=str, default="roneneldan/TinyStories-8M")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


MODEL_CONFIG = {
    'roneneldan/TinyStories-8M': {
        # https://huggingface.co/roneneldan/TinyStories-8M/blob/main/config.json
        **tiny_stories_8m_config,
        # max_length from TinyStories paper https://arxiv.org/pdf/2305.07759
        'ctx_len': 512
    }
}

def main():
    args = parse_args()

    # WandB run name
    name = f"{'Dense' if args.dense else 'Sparse'} TinyStories8M s={SEED}{' ' + args.tag if args.tag else ''}"
    dir_path = Path("data") / args.dataset.replace('/', '--') / name.replace(" ", "-") / "checkpoints"
    if dir_path.exists():
        dir_path = dir_path.parent / f"{dir_path.stem} {datetime.datetime.now().strftime('%y-%m-%d')}"

    if args.dense:
        sparse_batch_size_scalar = 1
    else:
        sparse_batch_size_scalar = 2

    # Effective batch size = 80 * 16 = 1280
    batch_size = 80 // sparse_batch_size_scalar
    gradient_accumulation_steps = 16 * sparse_batch_size_scalar
    
    # max_position_embeddings=context_length
    config = SparseGPTNeoConfig(**MODEL_CONFIG[args.config], sparse_mlp=not args.dense)
    model = SparseGPTNeoForCausalLM(config)
    
    tokenizer = AutoTokenizer.from_pretrained(args.dataset if not args.tokenizer else args.tokenizer)

    ptl_model = LightningWrapper(model, tokenizer, args.lr, betas=(args.b1, 0.95))
    ptl_model.cuda()

    name = f"{'Dense' if args.dense else 'Sparse'} 8M s={SEED} {args.tag + ' ' if args.tag else ''}"
    
    train_dataloader, val_dataloader  = get_dataloaders(
        args.dataset, tokenizer, batch_size, MODEL_CONFIG[args.config]['ctx_len']
    )
    
    wandb_logger = WandbLogger(project="tinystories", name=name) if not args.debug else None

    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path,
        save_top_k=-1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        deterministic=True,
        precision="bf16-mixed",
        accelerator="auto",
        max_epochs=args.max_epochs,
        devices=[0, 1, 2, 3, 4, 5, 6, 7] if not args.debug else [0],
        callbacks=[
            # threshold_callback,
            checkpoint_callback,
            EarlyStopping(
                monitor="val_loss", mode="min", patience=args.early_stopping_patience
            ),
        ],
        logger=wandb_logger if not args.debug else None,
        gradient_clip_val=1.0,
        accumulate_grad_batches=gradient_accumulation_steps,
    )

    trainer.fit(ptl_model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
