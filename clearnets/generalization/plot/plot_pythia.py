from matplotlib import legend
from pyparsing import line
import torch
from torch import Tensor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
import plotly.io as pio
import plotly.express as px

from clearnets.language.hf_client import get_model_checkpoints

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


def plot_singular_vals_across_time(
    images_path, checkpoint_data, log_checkpoints: list[int], 
    components=["mlp", "attn"], model_name: str = "pythia-160m"
):
    first_key = f"{components[0]}_singular_values"
    num_singular_values = len(checkpoint_data[log_checkpoints[0]][first_key][0])
    num_modules = len(checkpoint_data[log_checkpoints[0]][first_key])

    # Filter out all checkpoints except 5 evenly spaced ones including first and last
    log_checkpoints = log_checkpoints[::(len(log_checkpoints) // 4)]
    num_checkpoints = len(log_checkpoints)
    filtered_checkpoint_data = {key: checkpoint_data[key] for key in log_checkpoints}

    for component in components:
        key = f"{component}_singular_values"
        diff_key = f"{component}_diff_singular_values"
        name = f"{component.capitalize()} Singular Values"
        diff_name = f"{component.capitalize()} Diff Singular Values"

        subplot_titles = [f"Training step {step}" for step in log_checkpoints for _ in range(num_modules)]

        # Add num modules columns
        fig: go.Figure = make_subplots(
            rows=num_checkpoints,
            cols=num_modules, 
            shared_yaxes=True, 
            subplot_titles=subplot_titles,
            vertical_spacing=0.05,
            horizontal_spacing=0.01,
        ) # shared_xaxes=True, 
        
        # [checkpoints, modules, singular values]
        scores: list[list[list[float]]] = [scores[key] for scores in filtered_checkpoint_data.values()]
        diff_scores: list[list[list[float]]] = [scores[diff_key] for scores in filtered_checkpoint_data.values()]

        for row, (step, sc, d_sc) in enumerate(zip(log_checkpoints, scores, diff_scores), start=1):
            for col, (module_sc, module_d_sc) in enumerate(zip(sc, d_sc), start=1):
                fig.add_trace(go.Scatter(
                    x=list(range(num_singular_values)), y=module_sc, mode='lines', name=name, opacity=0.3,
                    showlegend=row==1 and col==1,
                    line=dict(color=px.colors.qualitative.Plotly[0])
                ), row=row, col=col)
                fig.add_trace(go.Scatter(
                    x=list(range(num_singular_values)), y=module_d_sc, mode='lines', name=diff_name, opacity=0.3,
                    showlegend=row==1 and col==1,
                    line=dict(color=px.colors.qualitative.Plotly[1])
                ), row=row, col=col)            

        
        fig.update_layout(
            title=f"{name} over Steps (Pythia-160M)",
            xaxis_title="Singular Value Index",
            yaxis_title="Singular Value",
            width=300 * num_modules,
            height = 250 * num_checkpoints,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.,
                xanchor="left",
                x=0.
            )
        )
        fig.write_image(images_path / f'{model_name.title()}_{component}_singular_values.pdf', format='pdf')

    
            

def add_seed_scores(
        fig, key: str, name: str, checkpoint_data, log_checkpoints: list[int],
        normalize: bool = False, color_idx: int = 0, line_dict=None,
        use_secondary_y = False,
    ):
    line_dict = (
        (line_dict | dict(color=px.colors.qualitative.Plotly[color_idx]))
        if line_dict is not None
        else dict(color=px.colors.qualitative.Plotly[color_idx])
    )
    
    if key not in checkpoint_data[log_checkpoints[0]].keys():
        print(f"Skipping {name} for seed because it does not exist in the checkpoint")

    
    scores: list[list[float]] = [scores[key] for scores in checkpoint_data.values()]
    # Reshape the scores to be a list of seeds (so the first element of each list is the first seed's list)
    module_scores = list(zip(*scores))

    # if normalize:
    #     # Normalize scores to be between 0 and 1
    #     max_score = max(scores)
    #     min_score = min(scores)
    #     scores = [(x - min_score) / (max_score - min_score) for x in scores]
    for scores_over_steps in module_scores:
        fig.add_trace(go.Scatter(
            x=log_checkpoints, y=scores_over_steps, mode='lines', name=name, opacity=0.3,
            line=line_dict,
            showlegend=False,
        ), secondary_y=use_secondary_y)

    # Get mean of seeds
    mean_scores = torch.tensor(module_scores, dtype=torch.float32).mean(dim=0)

    fig.add_trace(go.Scatter(
        x=log_checkpoints, y=mean_scores, mode='lines', name=name,
        line=line_dict
    ), secondary_y=use_secondary_y)


def plot_pythia(stop: int | None = None):
    model_name = "pythia-160m"
    out_path = Path(f"workspace/inference/{model_name}.pth")

    checkpoints = get_model_checkpoints(f"EleutherAI/{model_name}")    
    log2_keys = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16_000, 33_000, 66_000, 131_000, 143_000]
    log_checkpoints = {key: checkpoints[key] for key in log2_keys}

    stop = stop or len(log_checkpoints)

    checkpoint_data = torch.load(out_path)

    def add_scores(
            fig, key: str, name: str, normalize: bool = False,
            use_secondary_y=False
        ):
        if key not in checkpoint_data[list(log_checkpoints.keys())[0]].keys():
            print(f"Skipping {name} because it does not exist in the checkpoint")
            return
        
        scores = [scores[key] for scores in list(checkpoint_data.values())[:stop]]
        if isinstance(scores[0], Tensor):
            scores = [score.item() for score in scores]

        # Normalize scores to be between 0 and 1
        if normalize:
            max_score = max(scores)
            min_score = min(scores)
            scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        fig.add_trace(
            go.Scatter(x=list(log_checkpoints.keys()), y=scores, mode='lines', name=name), 
            secondary_y=use_secondary_y
        )

    images_path = Path("images")
    images_path.mkdir(exist_ok=True)
    
    # Add rank over checkpoints, with each module rank in the list being one seed and the main line being the average
    # data includes ["attn_layer_ranks"] and ["attn_layer_norms"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # get number of layers
    add_seed_scores(fig, f'attn_layer_ranks', f'Rank', checkpoint_data, log2_keys)
    # add_seed_scores(fig, f'attn_layer_norms', f'Norm', checkpoint_data, log2_keys)

    # Add horizontal line at attn_max_rank
    fig.add_hline(y=max(checkpoint_data[log2_keys[0]]['attn_max_rank']), line_dash="dash", line_color="black", line_width=1)

    fig.update_layout(
        title=f"Attention Layer Difference from Initialization Rank over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Rank",
        width=1000
    )
    fig.write_image(images_path / f'pythia_attn_layer_ranks.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_seed_scores(fig, f'mlp_layer_ranks', f'Rank', checkpoint_data, log2_keys)
    # add_seed_scores(fig, f'mlp_layer_norms', f'Norm', checkpoint_data, log2_keys)

    fig.add_hline(y=max(checkpoint_data[log2_keys[0]]['mlp_max_rank']), line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title=f"MLP Layer Difference from Initialization Rank over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Rank",
        width=1000
    )
    fig.write_image(images_path / f'pythia_mlp_layer_ranks.pdf', format='pdf')

    # Plot singular value spectrum with one line for diff and one line for regular
    plot_singular_vals_across_time(images_path, checkpoint_data, log2_keys)

    # Add SAE stuff
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores(fig, 'test_loss', 'Test loss', checkpoint_data)
    add_scores(fig, 'sae_entropy', f'H(SAE Nodes)', checkpoint_data, use_secondary_y=True
    )
    fig.update_layout(
        title=f"Loss and SAE Node Entropy over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_feature_sharing_entropy.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores(fig, 'test_loss', 'Test loss', checkpoint_data)

    add_scores(fig, 'hoyer', 'Hoyer', checkpoint_data, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and SAE Hoyer Norm over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_feature_sharing_hoyer.pdf', format='pdf')


    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores(fig, 'test_loss', 'Test loss', checkpoint_data)

    add_scores(fig, 'gini', 'Gini coefficient', checkpoint_data, use_secondary_y=True)
    fig.update_layout(
        title=f"Loss and Gini Coefficient of Mean SAE Activations over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_feature_sharing_gini.pdf', format='pdf')


    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])
    add_scores(fig, 'train_loss', 'Train loss', checkpoint_data)
    add_scores(fig, 'test_loss', 'Test loss', checkpoint_data)
    add_scores(fig, 'parameter_norm', 'Parameter norm', checkpoint_data)
    fig.update_layout(
        title=f"Loss and Parameter Norm over Steps (Pythia-160M)", 
        xaxis_title="Step", 
        yaxis_title="Loss",
        width=1000
    )                                           
    fig.write_image(images_path / f'pythia_parameter_norm.pdf', format='pdf')


if __name__ == "__main__":
    plot_pythia()