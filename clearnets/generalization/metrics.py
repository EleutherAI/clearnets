
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from scipy import stats
from scipy.optimize import linear_sum_assignment


def mean_l2(tensor: Tensor) -> float:
    return torch.linalg.vector_norm(tensor, ord=2, dim=1).mean().item()


def var_trace(tensor: Tensor) -> float:
    return tensor.var(dim=1).sum().item()


def abs_entropy(vec: ndarray):
    abs_vec = np.abs(vec)
    sum_scores = abs_vec.sum()
    probs = abs_vec / sum_scores

    return float(stats.entropy(probs))


def gini(vec: ndarray):
    n = len(vec)
    diffs = np.abs(np.subtract.outer(vec, vec)).mean()

    return float(np.sum(diffs) / (2 * n**2 * np.mean(vec)))


def hoyer(vec: ndarray):
    l1 = np.linalg.norm(vec, 1)
    l2 = np.linalg.norm(vec, 2)

    return float(l1 / l2)


def hoyer_square(vec: ndarray):
    return hoyer(vec ** 2)


def cosine_distance_matched_vectors(first: ndarray, second: ndarray):
    """
    Match vectors between two matrices using linear sum assignment with a cosine distance cost.
    
    Parameters:
    first: numpy array of shape (n, d) where n is number of vectors and d is the dimension
    second: numpy array of shape (m, d) where m is number of vectors and d is the dimension
    
    Returns:
    matches: list of tuples containing (first_index, second_index, similarity) for matched pairs
    """
    first_normalized = first / np.linalg.norm(first, axis=1)[:, np.newaxis]
    second_normalized = second / np.linalg.norm(second, axis=1)[:, np.newaxis]
    
    # Set cost to cosine distance between each pair of elements
    cost_matrix = 1 - np.dot(first_normalized, second_normalized.T)

    # Match elements using linear sum assignment to minimize cost
    row_idxs, col_idxs = linear_sum_assignment(cost_matrix)

    return row_idxs, col_idxs, cost_matrix[row_idxs, col_idxs]


def mean_matched_cosine_similarity(first: ndarray, second: ndarray) -> float:
    """
    Match vectors between two matrices using linear sum assignment with a cosine distance cost,
    then return the mean similarity across all matched pairs.
    
    Parameters:
    first: numpy array of shape (n, d) where n is number of vectors and d is the dimension
    second: numpy array of shape (m, d) where m is number of vectors and d is the dimension
    
    Returns:
    mean matched cosine similarity: float
    """
    row_ind, col_ind, cost_matrix = cosine_distance_matched_vectors(first, second)
    
    return 1 - cost_matrix[row_ind, col_ind].mean()

def network_compression(parameters: Tensor, parameters_initial: Tensor):    
    # TODO this might be wrong or inefficient, epistemic status: draft

    # SVD-based network compression seems like a good candidate for a generalisation measure
    # the weird thing might be choosing between SVD of the diff final - initial and SVD of the weight matrix itself
    # in the grokking regime with high weight decay you may want to do the weight matrix itself
    # if you were doing an actual encoding scheme you could just add an extra bit to indicate which you're using

    diff = parameters - parameters_initial
    rank = torch.linalg.matrix_rank(diff.float(), atol=1e-2).item()
    # print(rank)
    return rank


def singular_values(tensor: Tensor) -> list[float]:
    return torch.linalg.svdvals(tensor.float()).cpu().tolist()
