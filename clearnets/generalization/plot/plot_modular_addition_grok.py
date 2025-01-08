from pathlib import Path

from plotly.subplots import make_subplots
import torch
import plotly.graph_objects as go
import lovely_tensors as lt
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469

lt.monkey_patch()
device = torch.device("cuda")


def plot_modular_addition_grok():
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)

    # Load data
    checkpoint_datas = {}
    checkpoint_epochs = []
    
    seeds = [1, 2, 3, 4, 5] # 6, 7, 8
    for model_seed in seeds:
        run_identifier = f"{model_seed}"

        # Use existing on-disk model checkpoints
        import sys; sys.modules['clearnets.grok'] = sys.modules['clearnets']
        MODEL_PATH = Path(f"workspace/grok/{run_identifier}.pth")
        cached_data = torch.load(MODEL_PATH)
        
        OUT_PATH = Path(f"workspace/inference/{run_identifier}.pth")
        OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

        # Overwitten over loop
        checkpoint_epochs = cached_data["checkpoint_epochs"][::5]

        checkpoint_datas[model_seed] = torch.load(OUT_PATH)


    # Plot data
    def add_scores_if_exists(
            fig, key: str, score_type: str | None, name: str, 
            checkpoint_data, normalize: bool = False, use_secondary_y=False
    ):
        if score_type and score_type not in checkpoint_data[checkpoint_epochs[0]].keys():
            print(f"Skipping {name} for seed because {score_type} does not exist in the checkpoint")
        if (score_type and key not in checkpoint_data[checkpoint_epochs[0]][score_type].keys()) or (
            not score_type and key not in checkpoint_data[checkpoint_epochs[0]].keys()):
            print(f"Skipping {name} for seed because it does not exist in the checkpoint")
        
        scores = (
            [scores[score_type][key] for scores in checkpoint_data.values()]
            if score_type
            else [scores[key] for scores in checkpoint_data.values()]
        )

        # Normalize scores to be between 0 and 1
        if normalize:
            max_score = max(scores)
            min_score = min(scores)
            scores = [(x - min_score) / (max_score - min_score) for x in scores]

        fig.add_trace(go.Scatter(x=checkpoint_epochs, y=scores, mode='lines', name=name), secondary_y=use_secondary_y)


    def add_seed_scores_if_exists(
            fig, key: str, score_type: str | None, name: str, checkpoint_datas, 
            normalize: bool = False, color_idx: int = 0, line_dict=None,
            use_secondary_y = False
        ):
        line_dict = (
            (line_dict | dict(color=px.colors.qualitative.Plotly[color_idx]))
            if line_dict is not None
            else dict(color=px.colors.qualitative.Plotly[color_idx])
        )
        
        seed_scores = []
        for seed, checkpoint_data in checkpoint_datas.items():
            if score_type and score_type not in checkpoint_data[checkpoint_epochs[0]].keys():
                print(f"Skipping {name} for seed because {score_type} does not exist in the checkpoint")
            if (score_type and key not in checkpoint_data[checkpoint_epochs[0]][score_type].keys()) or (
                not score_type and key not in checkpoint_data[checkpoint_epochs[0]].keys()):
                print(f"Skipping {name} for seed because it does not exist in the checkpoint")
        
            scores = (
                [scores[score_type][key] for scores in checkpoint_data.values()]
                if score_type
                else [scores[key] for scores in checkpoint_data.values()]
            )

            if normalize:
                # Normalize scores to be between 0 and 1
                max_score = max(scores)
                min_score = min(scores)
                scores = [(x - min_score) / (max_score - min_score) for x in scores]

            seed_scores.append(scores)

            fig.add_trace(go.Scatter(
                x=checkpoint_epochs, y=scores, mode='lines', name=name, opacity=0.3,
                line=line_dict,
                showlegend=False,
            ), secondary_y=use_secondary_y)

        # Get mean of seeds
        mean_scores = torch.tensor(seed_scores, dtype=torch.float32).mean(dim=0)

        fig.add_trace(go.Scatter(
            x=checkpoint_epochs, y=mean_scores, mode='lines', name=name,
            line=line_dict
        ), secondary_y=use_secondary_y)
            
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', checkpoint_datas[seeds[0]])

    add_seed_scores_if_exists(
        fig, 
        'sae_entropy', 
        None, 
        f'H(SAE Nodes)', 
        checkpoint_datas,
        color_idx=5,
        use_secondary_y=True
    )
    fig.update_layout(
        title=f"Loss and SAE Node Entropy over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'grok_feature_sharing_entropy.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', checkpoint_datas[seeds[0]])

    add_seed_scores_if_exists(fig, 'hoyer', None, 'Hoyer', checkpoint_datas, color_idx=5, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and SAE Hoyer Norm over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'grok_feature_sharing_hoyer.pdf', format='pdf')


    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', checkpoint_datas[seeds[0]])

    add_seed_scores_if_exists(fig, 'gini', None, 'Gini coefficient', checkpoint_datas, color_idx=5, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and Gini Coefficient of Mean SAE Activations over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'grok_feature_sharing_gini.pdf', format='pdf')

    # add_seed_scores_if_exists(fig, 'hoyer_square', None, 'Hoyer Square', checkpoint_datas, color_idx=2)
    # add_seed_scores_if_exists(fig, 'sae_fvu', None, 'SAE FVU', checkpoint_datas, color_idx=3)
    
    
    OUT_PATH = Path(f"workspace/inference/1-wd=0.1,b2=0.95.pth")
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    baseline_data = torch.load(OUT_PATH)
    
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', baseline_data)
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', baseline_data)
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', baseline_data)
    add_scores_if_exists(fig, 'sae_entropy', None, f'H(SAE Nodes)', baseline_data, use_secondary_y=True)
    add_scores_if_exists(fig, 'hoyer', None, 'Hoyer', baseline_data, use_secondary_y=True)
    # add_seed_scores_if_exists(fig, 'hoyer_square', None, 'Hoyer Square', checkpoint_datas, color_idx=5)
    # Separate plot bc scale is quite different
    # add_scores_if_exists(fig, 'gini', None, 'Gini coefficient', baseline_data, use_secondary_y=True)
    # add_seed_scores_if_exists(fig, 'sae_fvu', None, 'SAE FVU', checkpoint_datas, color_idx=3)
    
    fig.update_layout(
        title=f"Loss, SAE Node Entropy, and SAE Activation Hoyer Norm over Epochs", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'grok_feature_sharing_baseline.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', baseline_data)
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', baseline_data)
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', baseline_data)
    add_scores_if_exists(fig, 'gini', None, 'Gini coefficient', baseline_data, use_secondary_y=True)
    
    fig.update_layout(
        title=f"Loss and Gini Coefficient of Mean SAE Activations over Epochs", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'grok_feature_sharing_baseline_gini.pdf', format='pdf')


if __name__ == "__main__":
    plot_modular_addition_grok()