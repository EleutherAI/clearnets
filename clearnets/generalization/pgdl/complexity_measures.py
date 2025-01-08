import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import ConvNextV2ForImageClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_X_y, _safe_indexing
from functools import partial
from torch.utils.data import DataLoader

# from clearnets.pgdl.intermediate_outputs import intermediateOutputs

def intermediate_outputs(batch: Tensor, model: ConvNextV2ForImageClassification):
    """
    Function to get intermediate outputs of a model

    Parameters
    ----------
    model : HF model
            The model for which the intermediate outputs are to be computed
    batch : torch.Tensor
            The input batch
    layer : int, optional
            The layer number from which to get the intermediate outputs

    Returns
    -------
    torch.Tensor
            The intermediate outputs
    """
    model.eval()
    with torch.no_grad():
        out = model(batch.to(model.device), output_hidden_states=True)
    return out.hidden_states



def get_generalization_score(model, dataloader: DataLoader, num_classes: int | None = None):
    complexity_db_score = complexityDB(model, dataloader, pool=True, compute_over=400)

    batches = [batch for batch in dataloader]
    if not num_classes:
        num_classes = batches[0]['label'].max().item() + 1
    complexity_mixup_score = complexityMixup(model, num_classes, batches)

    return complexity_db_score * (1 - complexity_mixup_score)



def complexityDB(
    model: ConvNextV2ForImageClassification,
    dataloader,
    pool=True,
    use_pca=False,
    layer: str | int = "initial",
    compute_over=400,
    batch_size=40,
    accumulate_batches=3,
):
    """
    Function to calculate feature clustering based measures. Based on the sklearn implementation of DB Index.

    Parameters
    ----------
    model : transformers model
            The model for which the complexity measure is to be computed
    dataset : data.Dataset
            Dataset object from PGDL data loader
    pool : bool, optional
            Whether to use max-pooling for dimensionality reduction, default True
    use_pca : bool, optional
            Whether to use PCA for dimensionality reduction, default False
    layer : str or int, optional
            Which layer to compute DB on. Either 'initial', for the first conv/pooling layer in the
            model, 'pre-penultimate' for the 3rd-from-last conv/pool layer, or an int indicating the
            layer. Defaults to 'initial'.

    Returns
    -------
    float
            complexity measure
    """

    assert accumulate_batches == 3, "TODO: implement accumulate_batches other than 3"
    def db(X, labels):
        X, labels = check_X_y(X, labels)
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        n_samples, _ = X.shape
        n_labels = len(le.classes_)

        assert 1 < n_labels < n_samples, (
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )

        intra_dists = np.zeros(n_labels)
        centroids = np.zeros((n_labels, len(X[0])), dtype=float)
        for k in range(n_labels):
            cluster_k = _safe_indexing(X, labels == k)
            centroid = cluster_k.mean(axis=0)
            centroids[k] = centroid
            intra_dists[k] = pairwise_distances(cluster_k, [centroid], metric="euclidean").mean()

        centroid_distances = pairwise_distances(centroids, metric="euclidean")

        if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
            return 0.0

        centroid_distances[centroid_distances == 0] = np.inf
        combined_intra_dists = intra_dists[:, None] + intra_dists
        scores = np.max(combined_intra_dists / centroid_distances, axis=1)
        return np.mean(scores)

    db_score = {}
    layer_dict = {"initial": [0, 1, 2], "pre-penultimate": [-3, -4, -5]}

    layer_list = layer_dict[layer] if isinstance(layer, str) else [layer]
    
    extractor = partial(intermediate_outputs, model=model) # mode = "pre"

    max_pool = nn.MaxPool2d(kernel_size=4, stride=4) if pool else nn.Identity()

    # layers = []

    # if isinstance(layer, str):
    #     for l in layer_dict[layer]:
    #         c = list(model.get_layer(index=l).get_config().keys())
    #         if "strides" in c:
    #             layers.append(l)
    #         if len(layers) == 1:
    #             break
    # else:
    #     for l in [layer]:
    #         c = list(model.get_layer(index=l).get_config().keys())
    #         if "strides" in c:
    #             layers.append(l)
    #         if len(layers) == 1:
    #             break

    for l in layer_list:
        accumulator = []
        for batch in dataloader:
            accumulator.append(batch)
            if len(accumulator) != accumulate_batches:
                continue
        
            batch1, batch2, batch3 = accumulator

            feature = np.concatenate(
                (
                    max_pool(extractor(batch1['input_ids'])[l])
                    .cpu().numpy()
                    .reshape(batch1['input_ids'].shape[0], -1),
                    max_pool(extractor(batch2['input_ids'])[l])
                    .cpu().numpy()
                    .reshape(batch2['input_ids'].shape[0], -1),
                    max_pool(extractor(batch3['input_ids'])[l])
                    .cpu().numpy()
                    .reshape(batch3['input_ids'].shape[0], -1),
                ),
                axis=0,
            )
            target = np.concatenate((batch1['label'], batch2['label'], batch3['label']), axis=0)
            if use_pca == True:
                pca = PCA(n_components=25)
                feature = pca.fit_transform(feature)
            try:
                db_score[l] += db(feature, target) / (compute_over // batch_size)
            except Exception as e:
                db_score[l] = db(feature, target) / (compute_over // batch_size)

            accumulator = []

    score = np.mean(list(db_score.values()))

    return score


def complexityMixup(model, n_classes, batches):
    """
    Function to calculate label-wise Mixup based measure

    Parameters
    ----------
    model : the HF model for which the complexity measure is to be computed
    dataloader : torch.utils.data.DataLoader object containing HF Dataset object

    Returns
    -------
    float
            complexity measure
    """

    def intrapolateImages(img, alpha=0.5):
        """
        Weighted combinations of pairs of images. 
        Input should be images of the same class.
        """
        temp = np.stack([img] * img.shape[0])
        tempT = np.transpose(temp, axes=(1, 0, 2, 3, 4))
        ret = alpha * temp + (1 - alpha) * tempT
        mask = np.triu_indices(img.shape[0], 1)
        return ret[mask]

    def veracityRatio(model, batches, label, version_loss=None, label_smoothing=0.1):
        results = []
        lossObject = nn.CrossEntropyLoss()
        for b in batches:
            img = b['input_ids'][b['label'] == label]
            lbl = b['label'][b['label'] == label]
            
            intra_img = intrapolateImages(img)
            intra_label = np.stack([label] * intra_img.shape[0])
            intra_logits = model(torch.from_numpy(intra_img).to(model.device)).logits

            if version_loss == "log":
                logLikelihood = lossObject(intra_label, intra_logits)
                results.append(logLikelihood)
            elif version_loss == "cosine":
                int_preds = F.softmax(intra_logits, dim=1)
                target = (
                    F.one_hot(torch.from_numpy(intra_label), int_preds.shape[-1]) * (1 - label_smoothing)
                    + label_smoothing / 2
                )
                results.append(
                    (nn.CosineSimilarity()(target, int_preds) + 1) / 2
                )
            elif version_loss == "mse":
                int_preds = F.softmax(intra_logits, dim=1)
                target = F.one_hot(
                    torch.from_numpy(intra_label), int_preds.shape[-1]
                )
                results.append(nn.MSELoss()(target, int_preds))
            else:
                # Compute accuracy
                int_preds = torch.argmax(intra_logits, dim=1)
                results.append((int_preds == label).float().mean().item())

        return np.mean(results)

    vr = []
    succeeded = []
    for l in range(n_classes):
        try:
            vr.append(veracityRatio(model, batches, l))
            succeeded.append(1)
        except Exception as e:
            print(e)
            succeeded.append(0)
            continue

    print(f"complexity measure with mixup succeeded for {sum(succeeded)} out of {n_classes} classes")

    return np.mean(vr)
