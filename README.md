# Clearnets

Development of and interpretability using disentangled architectures.

## Train

Contains code for training sparse feedforward transformers, regular transformers, SAEs, and Transcoders.

Train GptNeoX-style base transformers with either sparse or dense feedforwards:

```
python -m clearnets.train.train_transformer
python -m clearnets.train.train_transformer --dense
python -m clearnets.train.train_transformer --dense --dataset "lennart-finke/SimpleStories" --tokenizer "EleutherAI/gpt-neo-125m"
```

`sparse_gptneox` and `sparse_gptneox_config` are both modified from the GptNeoX implementation in HuggingFace transformers with a flag to enable training with sparse feedforwards.

Train a comparison SAE or transcoder on a dense feedforward transformer using:

```
python -m clearnets.train.train_sae_cli --run_name Transcoder-roneneldan--TinyStories-8M --ckpt "data/roneneldan--TinyStories/Dense-TinyStories8M-s=42-full-vocab/checkpoints/last.ckpt" --hookpoints transformer.h.*.mlp --ctx_len 512 --batch_size 128 --grad_acc_steps 2 --max_examples 768_000 --transcode
```

## Autointerp

Run:

```
python -m clearnets.autointerp.e2e_clearnet

```

with the following config arguments set to match the experiment:
- model
- sparse_model
- hookpoints
- mlp_mode

## Generalization

Experiments for tracking generalization of neural networks using disentangled activations.
