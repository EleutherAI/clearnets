

import torch.nn.functional as F
import pytorch_lightning as pl
from schedulefree import AdamWScheduleFree
from pytorch_lightning.callbacks import ModelCheckpoint

from clearnets.utils.utils import assert_type

class LogSpacedCheckpoint(ModelCheckpoint):
    """
    PyTorch Lightning callback that saves checkpoints at logarithmically spaced intervals.
    
    Args:
        num_checkpoints (int): Number of checkpoints to save over the training period
        save_last (bool): Whether to save the last checkpoint
        dirpath (str, optional): Directory to save checkpoints to
        filename (str, optional): Checkpoint filename format
    """
    def __init__(
        self,
        save_last: bool = True,
        dirpath: str = None,
        filename: str = None,
        **kwargs
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename if filename else "{epoch}",
            save_last=save_last,
            **kwargs
        )
        self.checkpoint_epochs = None
        
    def setup(self, trainer, pl_module, stage=None):
        super().setup(trainer, pl_module, stage)
        max_epochs = assert_type(int, trainer.max_epochs)
        
        powers = max_epochs.bit_length()
        self.checkpoint_epochs = sorted([2**i - 1 for i in range(powers)])
        print(self.checkpoint_epochs)

        
    def should_save_on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch in self.checkpoint_epochs:
            return True
        
        if self.save_last and trainer.current_epoch == trainer.max_epochs - 1:
            return True
        
        return False
    

class ScheduleFreeLightningWrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Hugging Face models that uses the AdamW schedule-free optimizer.
    """
    def __init__(self, model, learning_rate=1e-4, betas=(0.9, 0.999), warmup_steps=10_000):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.betas = betas
        self.warmup_steps = warmup_steps
        
    def forward(self, pixel_values):
        return self.model(pixel_values).logits

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
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = self.model(x).logits
        loss = F.cross_entropy(y, batch[1])
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = self.model(x).logits
        loss = F.cross_entropy(y, batch[1])
        acc = (y.argmax(dim=-1) == batch[1]).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = self.model(x).logits
        loss = F.cross_entropy(y, batch[1])
        acc = (y.argmax(dim=-1) == batch[1]).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

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
    
    def configure_optimizers(self):
        self.optimizer = AdamWScheduleFree(
            self.parameters(), 
            lr=self.learning_rate, 
            betas=self.betas, 
            warmup_steps=self.warmup_steps
        )

        return {
            "optimizer": self.optimizer,
        }
    