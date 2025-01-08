# Initial modular addition experiments using many kinds of patching
# And both residual and SAE node scores
# Kept for reference
from pathlib import Path

import torch
import plotly.graph_objects as go
import lovely_tensors as lt
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469

lt.monkey_patch()
device = torch.device("cuda")


def main():
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)

    checkpoint_datas = {}
    checkpoint_epochs = []
    
    seeds = [1, 2, 3, 4, 5] # 6, 7, 8
    for model_seed in seeds:
        run_identifier = f"{model_seed}"

        # Use existing on-disk model checkpoints
        MODEL_PATH = Path(f"workspace/grok/{run_identifier}.pth")
        cached_data = torch.load(MODEL_PATH)
        
        OUT_PATH = Path(f"workspace/inference/{run_identifier}.pth")
        OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

        # Overwitten over loop
        checkpoint_epochs = cached_data["checkpoint_epochs"][::5]

        checkpoint_datas[model_seed] = torch.load(OUT_PATH)

    def add_scores_if_exists(
            fig, key: str, score_type: str | None, name: str, 
            checkpoint_data, normalize: bool = True, use_secondary_y=False
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
            # scores = [score / max_score for score in scores]

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
            
    from plotly.subplots import make_subplots
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])

    # fig = go.Figure()
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Underlying model and task stats
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', checkpoint_datas[seeds[0]], normalize=False)
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', checkpoint_datas[seeds[0]], normalize=False)
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', checkpoint_datas[seeds[0]], normalize=False)

    # for idx, node_score_type in enumerate([
    #     'sae_ig_zero', 'sae_attrib_zero', 'sae_grad_zero', 'sae_ig_patch', 'sae_attrib_patch'
    # ], start=3): 
    #     ablation = '0' if 'zero' in node_score_type else 'patch'
    #     type = 'IG' if 'ig' in node_score_type else 'AtP' if 'attrib' in node_score_type else 'Gradient'

    # add_seed_scores_if_exists(
    #     fig, 
    #     'entropy', 
    #     node_score_type, 
    #     f'H(SAE Nodes) {ablation} ablation, {type}', 
    #     checkpoint_datas,
    #     color_idx=idx
    # )

    add_seed_scores_if_exists(
        fig, 
        'sae_entropy', 
        None, 
        f'H(SAE Nodes)', 
        checkpoint_datas,
        color_idx=5,
        use_secondary_y=True
    )
    # add_seed_scores_if_exists(fig, 'hoyer', None, 'Hoyer', checkpoint_datas, color_idx=5, use_secondary_y=True)
    # add_seed_scores_if_exists(fig, 'hoyer_square', None, 'Hoyer Square', checkpoint_datas, color_idx=2)
    # add_seed_scores_if_exists(fig, 'gini', None, 'Gini coefficient', checkpoint_datas, color_idx=5, use_secondary_y=True)
    # add_seed_scores_if_exists(fig, 'sae_fvu', None, 'SAE FVU', checkpoint_datas, color_idx=3)
    
    fig.update_layout(
        title=f"Grokking Loss and SAE Node Entropy over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        yaxis2_title="Sparsity score",
        # yaxis_title="Loss (log scale)",
        # yaxis_type="log",
        width=1000
    )                                           
    fig.write_image(images_path / f'feature_sharing_entropy.pdf', format='pdf')


    from plotly.subplots import make_subplots
    fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])

    # fig = go.Figure()
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Underlying model and task stats
    OUT_PATH = Path(f"workspace/inference/1-wd=0.1,b2=0.95.pth")
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    baseline_data = torch.load(OUT_PATH)
    
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', baseline_data, normalize=False)
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', baseline_data, normalize=False)
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', baseline_data, normalize=False)
    add_scores_if_exists(
        fig, 
        'sae_entropy', 
        None, 
        f'H(SAE Nodes)', 
        baseline_data,
        # color_idx=3,
        use_secondary_y=True
    )
    add_scores_if_exists(fig, 'hoyer', None, 'Hoyer', baseline_data, use_secondary_y=True, normalize=False) # color_idx=4, 
    # add_seed_scores_if_exists(fig, 'hoyer_square', None, 'Hoyer Square', checkpoint_datas, color_idx=5)
    add_scores_if_exists(fig, 'gini', None, 'Gini coefficient', baseline_data, use_secondary_y=True, normalize=False) # color_idx=5, 
    # add_seed_scores_if_exists(fig, 'sae_fvu', None, 'SAE FVU', checkpoint_datas, color_idx=3)
    
    fig.update_layout(
        title=f"Grokking Loss and SAE Node Entropy over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        # yaxis_title="Loss (log scale)",
        # yaxis_type="log",
        # Label right y axis as "Sparsity score"
        yaxis2_title="Sparsity score",
        width=1000
    )                                           
    fig.write_image(images_path / f'feature_sharing_baseline.pdf', format='pdf')

    exit(0)


    fig = go.Figure()
    # Underlying model and task stats
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', checkpoint_datas[seeds[0]])

    for idx, node_score_type in enumerate([
        'residual_ig_zero', 'residual_attrib_zero', 'residual_grad_zero', 'residual_ig_patch', 'residual_attrib_patch'
    ], start=3):
        ablation = '0' if 'zero' in node_score_type else 'patch'
        type = 'IG' if 'ig' in node_score_type else 'AtP' if 'attrib' in node_score_type else 'Gradient'

        add_seed_scores_if_exists(
            fig, 
            'entropy', 
            node_score_type, 
            f'H(SAE Nodes) {ablation} ablation, {type}', 
            checkpoint_datas,
            color_idx=idx,
        )
    
    fig.update_layout(
        title=f"Grokking Loss and Residual Node Entropy over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        width=1000
    )                                           
    fig.write_image(images_path / f'node_resid_entropies.pdf', format='pdf')


    fig = go.Figure()
    
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', checkpoint_datas[seeds[0]])

    for idx, node_score_type in enumerate([
        'residual_ig_zero', 'residual_attrib_zero', 'residual_ig_patch', 'residual_attrib_patch'
    ], start=3): # excluding 'residual_grad_zero' beause it's pure noise and out of scale
        ablation = '0' if 'zero' in node_score_type else 'patch'
        type = 'IG' if 'ig' in node_score_type else 'AtP' if 'attrib' in node_score_type else 'Gradient'

        add_seed_scores_if_exists(
            fig, 
            'linearized_circuit_feature_count', 
            node_score_type, 
            f'Proxy circuit size, {ablation}, {type}', 
            checkpoint_datas,
            color_idx=idx,
            line_dict=dict(dash='dash')
        )

        add_seed_scores_if_exists(
            fig, 
            'circuit_feature_count', 
            node_score_type, 
            f'Circuit size, {ablation}, {type}', 
            checkpoint_datas,
            color_idx=idx,
        )

    fig.update_layout(
        title=f"Grokking Loss and Residual Circuit Size over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        width=1000
    )                                           
    fig.write_image(images_path / f'node_counts_residual.pdf', format='pdf')


    fig = go.Figure()
    
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', checkpoint_datas[seeds[0]])

    for idx, node_score_type in enumerate([ # 'residual_attrib_zero', 'residual_attrib_patch'
        'residual_ig_zero', 'residual_ig_patch', 
    ], start=3): # excluding 'residual_grad_zero' beause it's pure noise and out of scale
        ablation = '0' if 'zero' in node_score_type else 'patch'
        type = 'IG' if 'ig' in node_score_type else 'AtP' if 'attrib' in node_score_type else 'Gradient'

        add_seed_scores_if_exists(
            fig, 
            'linearized_circuit_feature_count', 
            node_score_type, 
            f'Proxy circuit size, {ablation}, {type}', 
            checkpoint_datas,
            color_idx=idx,
            line_dict=dict(dash='dash')
        )

        # if 'ig' in node_score_type and 'patch' in node_score_type:
        #     continue

        # add_seed_scores_if_exists(
        #     fig, 
        #     'circuit_feature_count', 
        #     node_score_type, 
        #     f'Circuit size, {ablation}, {type}', 
        #     checkpoint_datas,
        #     color_idx=idx,
        # )

    fig.update_layout(
        title=f"Grokking Loss and Residual Circuit Size over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        width=1000
    )                                           
    fig.write_image(images_path / f'node_counts_residual_cherrypicked.pdf', format='pdf')


    # Node count SAE AtP
    fig = go.Figure()
    
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', checkpoint_datas[seeds[0]])

    sae_keys = ['sae_attrib_zero'] #'sae_ig_zero',  'sae_ig_patch', 'sae_attrib_patch', 'sae_grad_zero'
    for idx, node_score_type in enumerate(sae_keys, start=3):
        ablation = '0' if 'zero' in node_score_type else 'patch'
        type = 'IG' if 'ig' in node_score_type else 'AtP' if 'attrib' in node_score_type else 'Gradient'

        add_seed_scores_if_exists(
            fig, 
            'linearized_circuit_feature_count', 
            node_score_type, 
            f'Proxy circuit size, {ablation}, {type}', 
            checkpoint_datas,
            color_idx=idx
        )

        add_seed_scores_if_exists(
            fig, 
            'circuit_feature_count', 
            node_score_type, 
            f'Circuit size, {ablation}, {type}', 
            checkpoint_datas,
            color_idx=idx,
            line_dict=dict(dash='dash')
        )

    fig.update_layout(
        title=f"Grokking Loss and SAE Circuit Size over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        width=1000
    )                                           
    fig.write_image(images_path / f'node_counts_sae_atp.pdf', format='pdf')

    # Node count SAE IG
    fig = go.Figure()
    
    add_scores_if_exists(fig, 'train_loss', None, 'Train loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'test_loss', None, 'Test loss', checkpoint_datas[seeds[0]])
    add_scores_if_exists(fig, 'parameter_norm', None, 'Parameter norm', checkpoint_datas[seeds[0]])

    sae_keys = ['sae_ig_zero'] # 'sae_attrib_zero', 'sae_ig_patch', 'sae_attrib_patch', 'sae_grad_zero'
    for idx, node_score_type in enumerate(sae_keys, start=3):
        ablation = '0' if 'zero' in node_score_type else 'patch'
        type = 'IG' if 'ig' in node_score_type else 'AtP' if 'attrib' in node_score_type else 'Gradient'

        add_seed_scores_if_exists(
            fig, 
            'linearized_circuit_feature_count', 
            node_score_type, 
            f'Proxy circuit size, {ablation}, {type}', 
            checkpoint_datas,
            color_idx=idx
        )

        add_seed_scores_if_exists(
            fig, 
            'circuit_feature_count', 
            node_score_type, 
            f'Circuit size, {ablation}, {type}', 
            checkpoint_datas,
            color_idx=idx,
            line_dict=dict(dash='dash')
        )

    fig.update_layout(
        title=f"Grokking Loss and SAE Circuit Size over Epochs for {len(seeds)} seeds", 
        xaxis_title="Epoch", 
        yaxis_title="Loss",
        width=1000
    )                                           
    fig.write_image(images_path / f'node_counts_sae_ig.pdf', format='pdf')



    # SAE summary stats
    # add_scores_if_exists(fig, 'sae_loss', 'SAE loss') 
    # add_scores_if_exists(fig, 'sae_mse', 'SAE MSE')


    


if __name__ == "__main__":
    main()