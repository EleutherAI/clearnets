# Clearnets

Development of and interpretability using disentangled architectures.

## Train

Contains code for training sparse feedforward transformers, regular transformers, SAEs, and Transcoders.

## Autointerp

Contains scripts for calling the sae-auto-interp code on trained models.

### Notes

train_tinystories_transformers is the entry point for the project. Use it to train GptNeoX-style transformers with either sparse or dense feedfowards. sparse_gptneox and sparse_gptneox_config are both modified from the GptNeoX implementation in HuggingFace transformers with a flag that enables training with sparse feedforwards.

Train a comparison SAE or transcoder on a dense feedfwd transformer using train_sae_cli.

Evaluate the disengtanglement using populate_autointerp_cache_sae, populate_autointerp_cache_sparse and autointerp_explain, all modified from a script provided by Gonçalo to run sae-auto-interp.

Example:

```
python -m clearnets.autointerp.populate_cache_sae --dataset_repo "roneneldan/TinyStories" --dataset_split "train[:2%]" --dataset_row "text" --n_tokens 10_000_000

python -m clearnets.autointerp.populate_autointerp_cache_sparse --dataset_repo "roneneldan/TinyStories" --dataset_split "train[:2%]" --dataset_row "text" --n_tokens 10_000_000

python -m clearnets.autointerp.autointerp_explain --model "data/tinystories/sparse-8m-max-e=200-esp=15-s=42" --modules "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.0.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.1.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.2.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.3.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.4.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.5.mlp" "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.7.mlp" --n_random 50 --n_examples_test 50 --train_type "quantiles" --n_examples_train 40 --n_quantiles 10 --width 8192

python -m clearnets.autointerp.autointerp_explain --model "data/tinystories/dense-8m-max-e=200-esp=15-s=42" --modules "roneneldan/TinyStories/SAE/.transformer.h.0.mlp" "roneneldan/TinyStories/SAE/.transformer.h.1.mlp" "roneneldan/TinyStories/SAE/.transformer.h.2.mlp" "roneneldan/TinyStories/SAE/.transformer.h.3.mlp" "roneneldan/TinyStories/SAE/.transformer.h.4.mlp" "roneneldan/TinyStories/SAE/.transformer.h.5.mlp" "roneneldan/TinyStories/SAE/.transformer.h.6.mlp" "roneneldan/TinyStories/SAE/.transformer.h.7.mlp" --n_random 50 --n_examples_test 50 --train_type "quantiles" --n_examples_train 40 --n_quantiles 10 --width 8192

python -m clearnets.autointerp.autointerp_plot

```

## Generalization

Experiments for tracking generalization of neural networks using disentangled activations.