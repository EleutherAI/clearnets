from argparse import ArgumentParser
from pathlib import Path
import datetime
import os

from datasets import load_dataset, DatasetDict
from sae.data import chunk_and_tokenize
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback, GPTNeoXTokenizerFast
import torch
import torch.distributed as dist

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from clearnets.train.sparse_gptneox import SparseGPTNeoXForCausalLM
from clearnets.train.sparse_gptneox_config import SparseGPTNeoXConfig
from clearnets.utils import set_seeds, assert_type

torch.set_float32_matmul_precision("high")
SEED = 42
set_seeds(SEED)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/FineWeb-restricted")
    parser.add_argument("--config", type=str, default="FineWeb-28M")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128) # dense batch size 128, sparse batch size
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


MODEL_CONFIG = {
    'FineWeb-28M': {
        # Use default config values
        # max_length from TinyStories paper https://arxiv.org/pdf/2305.07759
        'ctx_len': 512
    }
}

class GenerationLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer, model, eval_dataset, log_interval=100):
        self.tokenizer = tokenizer
        self.model = model
        self.eval_dataset = eval_dataset
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_interval == 0:
            # Generate output
            with torch.no_grad():
                output = self.model.generate(max_length=50)
            
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Step {state.global_step}:")
            print(f"Generated: {generated_text}\n")


def main():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.dataset if not args.tokenizer else args.tokenizer)

    dataset = assert_type(
        DatasetDict,
        load_dataset(args.dataset, name="sample-10BT" if args.dataset == "HuggingFaceFW/fineweb" else None)
    )

    processed_dataset = chunk_and_tokenize(
        dataset,
        tokenizer,
        max_seq_len=512,
        text_key="text" if "text" in dataset["train"].column_names else "story",
    )
    processed_dataset = assert_type(DatasetDict, processed_dataset)
    processed_dataset.set_format(type="torch", columns=["input_ids"])
    processed_dataset = processed_dataset['train'].train_test_split(test_size=0.005, seed=42)

    # WandB run name
    name = f"{'Dense' if args.dense else 'Sparse'} FineWeb10B-28M s={SEED}{' ' + args.tag if args.tag else ''}"
    dir_path = Path("data") / args.dataset.replace('/', '--') / name.replace(" ", "-") / "checkpoints"
    if dir_path.exists():
        dir_path = dir_path.parent / f"{dir_path.stem} {datetime.datetime.now().strftime('%y-%m-%d')}"
 
    config = SparseGPTNeoXConfig(**MODEL_CONFIG[args.config], sparse_mlp=not args.dense)
    model = SparseGPTNeoXForCausalLM(config)

    name = f"{'Dense' if args.dense else 'Sparse'} 3M s={SEED} {args.tag + ' ' if args.tag else ''}"
    training_args = TrainingArguments(
        adam_beta1=args.b1,
        adam_beta2=0.95,
        bf16=True,
        ddp_find_unused_parameters=False,
        eval_strategy="epoch",     # evaluate at the end of each epoch
        learning_rate=args.lr,
        logging_dir="./logs",            # directory for TensorBoard logs
        logging_steps=100,
        num_train_epochs=args.num_epochs,              # number of epochs
        optim="schedule_free_adamw",
        output_dir=dir_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size, # dense batch size 512, sparse batch size 384
        save_strategy="epoch",           # save model checkpoints each epoch
    )

    trainer = Trainer(
        args=training_args,
        callbacks=[GenerationLoggerCallback(tokenizer, model, processed_dataset['test'])],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        eval_dataset=processed_dataset['test'],
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_dataset['train'],
    )
    trainer.train()


if __name__ == "__main__":
    main()
