
# Probably superseded by cifar10 vision which runs inference on three models and a harder dataset
from pathlib import Path

from plotly.subplots import make_subplots
import torch
import plotly.graph_objects as go
import lovely_tensors as lt
import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
lt.monkey_patch()


def plot_mnist_vit(model_path: Path, out_path: Path, images_path: Path):
    checkpoint_data = torch.load(out_path)
    checkpoint_epochs = torch.load(model_path)['checkpoint_epochs']

    def add_scores_if_exists(fig, key: str, name: str, normalize: bool = False, use_secondary_y=False):
        if key not in checkpoint_data[checkpoint_epochs[0]].keys():
            print(f"Skipping {name} because it does not exist in the checkpoint")
            return

        scores = [scores[key] for scores in checkpoint_data.values()]
        if normalize:
            # Normalize scores to be between 0 and 1
            max_score = max(scores)
            min_score = min(scores)
            scores = [(score - min_score) / (max_score - min_score) for score in scores]

        fig.add_trace(go.Scatter(x=checkpoint_epochs, y=scores, mode='lines', name=name), secondary_y=use_secondary_y)

    
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    
    add_scores_if_exists(fig, 'train_loss', 'Train loss', normalize=False)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', normalize=False)
    add_scores_if_exists(fig, 'parameter_norm', 'Parameter norm')
    add_scores_if_exists(fig, 'sae_entropy', f'H(SAE Nodes)', use_secondary_y=True)
    add_scores_if_exists(fig, 'test_sae_entropy', 'Test H(SAE Nodes)', False, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and SAE Node Entropy over Epochs (MNIST)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'mnist_feature_sharing_entropy.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', False)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', False)

    add_scores_if_exists(fig, 'hoyer', 'Hoyer', False, use_secondary_y=True)
    add_scores_if_exists(fig, 'test_hoyer', 'Test Hoyer', False, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and SAE Hoyer Norm over Epochs (MNIST)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'mnist_feature_sharing_hoyer.pdf', format='pdf')


    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', 'Train loss', False)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', False)

    add_scores_if_exists(fig, 'gini', 'Gini coefficient', False, use_secondary_y=True)
    add_scores_if_exists(fig, 'test_gini', 'Test Gini', False, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and Gini Coefficient of Mean SAE Activations over Epochs (mnist)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'mnist_feature_sharing_gini.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    
    add_scores_if_exists(fig, 'train_loss', 'Train loss', normalize=False)
    add_scores_if_exists(fig, 'test_loss', 'Test loss', normalize=False)
    add_scores_if_exists(fig, 'parameter_norm', 'Parameter norm')
    add_scores_if_exists(fig, 'mean_l2', 'Mean L2', use_secondary_y=True)
    add_scores_if_exists(fig, 'test_mean_l2', 'Test Mean L2', use_secondary_y=True)
    add_scores_if_exists(fig, 'var_l2', 'Variance L2', use_secondary_y=True)
    add_scores_if_exists(fig, 'test_var_l2', 'Test Variance L2', use_secondary_y=True)
    add_scores_if_exists(fig, 'skew_l2', 'Skewness L2', use_secondary_y=True)
    add_scores_if_exists(fig, 'test_skew_l2', 'Test Skewness L2', use_secondary_y=True)
    add_scores_if_exists(fig, 'kurt_l2', 'Kurtosis L2', use_secondary_y=True)
    add_scores_if_exists(fig, 'test_kurt_l2', 'Test Kurtosis L2', use_secondary_y=True)
    add_scores_if_exists(fig, 'var_trace', 'Variance trace', use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and Stats over Epochs (MNIST)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'mnist_stats.pdf', format='pdf')
