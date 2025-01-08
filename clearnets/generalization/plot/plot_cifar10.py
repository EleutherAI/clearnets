from pathlib import Path
from typing import Any
from plotly.subplots import make_subplots
import torch
import plotly.graph_objects as go
import lovely_tensors as lt
import plotly.io as pio
from pprint import pprint

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
lt.monkey_patch()

# Incubating: Collect rank, margins, original PGDL generalization score over training. Can we predict generalization in advance?
# Now: plot rank over layers for each model in one plot
# Plot parameter norm for comparison
# print generalization score for each model

# I have the singular values for select modules through 3 models
# Plot singular values for each model in one plot

def plot_singular_vals(
    images_path, inference_data,
):
    # Add num modules columns
    subplot_titles = list(inference_data.keys())
    num_modules = max(len(inference_data[title]["mlp_singular_values"]) for title in subplot_titles)
    fig: go.Figure = make_subplots(
        rows=3,
        cols=num_modules, 
        shared_yaxes=True, 
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.01,
    ) 

    first_key = f"{components[0]}_singular_values"
    num_singular_values = len(data[model_name][first_key][0])
    num_modules = len(data[model_name][first_key])


    for model_name, model_data in inference_data.items():
        components = model_data["components"]
        for component in components:
            key = f"{component}_singular_values"
            diff_key = f"{component}_diff_singular_values"
            name = f"{component.capitalize()} Singular Values"
            diff_name = f"{component.capitalize()} Diff Singular Values"

        

        
        
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

    

def plot(out_path: Path, images_path: Path):
    # model_name -> metric_name -> list[float] | float | list[list[float]]
    inference_data: dict[str, dict[str, Any]] = torch.load(out_path)

    # attach component info to inference data
    for model_name in inference_data.keys():
        if model_name == "ViT" or model_name == "Swin":
            inference_data[model_name]["components"] = ["mlp", "attn"]
        elif model_name == "ConvNext":
            inference_data[model_name]["components"] = ["conv"]

    plot_singular_vals(images_path, inference_data)

    def add_scores_if_exists(fig, scores, key: str, name: str, normalize: bool = False, use_secondary_y=False, col: int = 1, model_name: str = ""):
        if normalize:
            # Normalize scores to be between 0 and 1
            max_score = max(scores)
            min_score = min(scores)
            scores = [(score - min_score) / (max_score - min_score) for score in scores]

        fig.add_trace(go.Scatter(x=list(range(len(scores))), y=scores, mode='lines', name=name), secondary_y=use_secondary_y, row=1, col=col)
        if model_name:
            fig.update_xaxes(title_text=model_name, row=1, col=col)

    
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=3)
    for col, (model_name, model_data) in enumerate(inference_data.items(), start=1):
        add_scores_if_exists(fig, model_data['train_loss'], 'train_loss', 'Train loss', normalize=False, col=col)
        add_scores_if_exists(fig, model_data['test_loss'], 'test_loss', 'Test loss', normalize=False, col=col)
        add_scores_if_exists(fig, model_data['parameter_norm'], 'parameter_norm', 'Parameter norm', col=col)    
        add_scores_if_exists(fig, model_data['sae_entropy'], 'sae_entropy', f'H(SAE Nodes)', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['test_sae_entropy'], 'test_sae_entropy', 'Test H(SAE Nodes)', False, use_secondary_y=True, col=col)

    fig.update_layout(
        title=f"Loss and SAE Node Entropy over Epochs (MNIST)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'CIFAR-10_feature_sharing_entropy.pdf', format='pdf')

    # plot_singular_vals(images_path, inference_data, model_data, components=["mlp", "attn"], model_name=model_name)

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=3)
    for col, (model_name, model_data) in enumerate(inference_data.items(), start=1):
        add_scores_if_exists(fig, model_data['hoyer'], 'hoyer', 'Hoyer', False, use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['test_hoyer'], 'test_hoyer', 'Test Hoyer', False, use_secondary_y=True, col=col)
    fig.update_layout(
        title=f"Loss and SAE Hoyer Norm over Epochs (CIFAR-10)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'CIFAR-10_feature_sharing_hoyer.pdf', format='pdf')


    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=3)
    for col, (model_name, model_data) in enumerate(inference_data.items(), start=1):
        add_scores_if_exists(fig, model_data['gini'], 'gini', 'Gini coefficient', False, use_secondary_y=True, col=col, model_name=model_name)
        add_scores_if_exists(fig, model_data['test_gini'], 'test_gini', 'Test Gini', False, use_secondary_y=True, col=col, model_name=model_name)
    fig.update_layout(
        title=f"Loss and Gini Coefficient of Mean SAE Activations over Epochs (CIFAR-10)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'CIFAR-10_feature_sharing_gini.pdf', format='pdf')

    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]], rows=1, cols=3)
    for col, (model_name, model_data) in enumerate(inference_data.items(), start=1):
        add_scores_if_exists(fig, model_data['train_loss'], 'train_loss', 'Train loss', normalize=False, col=col)
        add_scores_if_exists(fig, model_data['test_loss'], 'test_loss', 'Test loss', normalize=False, col=col)
        add_scores_if_exists(fig, model_data['parameter_norm'], 'parameter_norm', 'Parameter norm', col=col)
        add_scores_if_exists(fig, model_data['mean_l2'], 'mean_l2', 'Mean L2', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['test_mean_l2'], 'test_mean_l2', 'Test Mean L2', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['var_l2'], 'var_l2', 'Variance L2', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['test_var_l2'], 'test_var_l2', 'Test Variance L2', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['skew_l2'], 'skew_l2', 'Skewness L2', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['test_skew_l2'], 'test_skew_l2', 'Test Skewness L2', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['kurt_l2'], 'kurt_l2', 'Kurtosis L2', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['test_kurt_l2'], 'test_kurt_l2', 'Test Kurtosis L2', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['var_trace'], 'var_trace', 'Variance trace', use_secondary_y=True, col=col)
        add_scores_if_exists(fig, model_data['test_var_trace'], 'test_var_trace', 'Test Variance trace', use_secondary_y=True, col=col)
    fig.update_layout(
        title=f"Loss and Stats over Epochs (CIFAR-10)", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'CIFAR-10_stats.pdf', format='pdf')



if __name__ == "__main__":
    out_path = Path('data') / 'vision_cifar10' / 'inference_data.pt'
    images_path = Path("images")
    
    out_path.parent.mkdir(exist_ok=True, parents=True)
    images_path.mkdir(exist_ok=True)
    plot(out_path, images_path)