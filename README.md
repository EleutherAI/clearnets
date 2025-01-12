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

Contains scripts for evaluating trained models using sae-auto-interp.

First, populate the cache with activations from a trained model or its auxiliary SAE/transcoder:

```
python -m clearnets.autointerp.populate_cache_sae --dataset_repo "roneneldan/TinyStories" --dataset_split "train[:2%]" --dataset_row "text" --n_tokens 10_000_000 --model_ckpt "data/roneneldan--TinyStories/Dense-TinyStories8m-s=42-full-vocab/checkpoints/last.ckpt" --sae_dir "data/sae/Dense TinyStories8M Transcoder 32x 8192 s=42 epoch 21"

python -m clearnets.autointerp.populate_autointerp_cache_sparse --dataset_repo "roneneldan/TinyStories" --dataset_split "train[:2%]" --dataset_row "text" --n_tokens 10_000_000 --model "Dense-TinyStories8m-s=42-full-vocab" --epoch 6
```

Then run an autointerp pipeline and visualize the results:

```
python -m clearnets.autointerp.autointerp_explain --model "roneneldan--TinyStories/Sparse-TinyStories8M-s=42-full-vocab" --modules "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.0.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.1.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.2.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.3.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.4.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.5.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.7.mlp" --n_random 50 --n_examples_test 50 --train_type "quantiles" --n_examples_train 40 --n_quantiles 10 --width 8192

python -m clearnets.autointerp.autointerp_explain --model "roneneldan--TinyStories/Dense-TinyStories8M-s=42-full-vocab" --modules "roneneldan/TinyStories/SAE/.transformer.h.0.mlp" "roneneldan/TinyStories/SAE/.transformer.h.1.mlp" "roneneldan/TinyStories/SAE/.transformer.h.2.mlp" "roneneldan/TinyStories/SAE/.transformer.h.3.mlp" "roneneldan/TinyStories/SAE/.transformer.h.4.mlp" "roneneldan/TinyStories/SAE/.transformer.h.5.mlp" "roneneldan/TinyStories/SAE/.transformer.h.6.mlp" "roneneldan/TinyStories/SAE/.transformer.h.7.mlp" --n_random 50 --n_examples_test 50 --train_type "quantiles" --n_examples_train 40 --n_quantiles 10 --width 8192

python -m clearnets.autointerp.autointerp_plot

```

## Generalization

Experiments for tracking generalization of neural networks using disentangled activations.