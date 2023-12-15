import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
import seaborn as sns
# Function to load model weights
def load_model_weights(model_path):
    return torch.load(model_path)


# Function to calculate similarity between two sets of weights
def _calculate_weight_similarity(weights1, weights2, similarity_metric='cosine', last_n_layers=0, baseline_weights=None):
    similarities = []
    keys = list(weights1.keys())
    for key in keys[-last_n_layers:]:
        if key in weights2:
            w1 = weights1[key].flatten().detach().numpy()
            w2 = weights2[key].flatten().detach().numpy()
            if baseline_weights is not None:
                w0 = baseline_weights[key].flatten().detach().numpy()
                w1 = w1 - w0
                w2 = w2 - w0
            if similarity_metric == 'cosine':
                similarity = 1 - cosine(w1, w2)
            elif similarity_metric == 'euclidean':
                similarity = 1 / (1 + euclidean(w1, w2))
            else:
                raise ValueError('Invalid similarity metric')
            similarities.append(similarity)
    overall_similarity = np.mean(similarities)
    return overall_similarity



def calculate_weight_similarity(weights_list, similarity_metric='cosine', last_n_layers=0, baseline_weights=None):
    similarities = np.ones((len(weights_list), len(weights_list)))
    for i in range(len(weights_list)):
        for j in range(i+1, len(weights_list)):
            similarities[i][j] = _calculate_weight_similarity(weights_list[i], weights_list[j], similarity_metric, last_n_layers, baseline_weights)
            similarities[j][i] = similarities[i][j]
    return similarities


def plot_weight_similarity(similarities, vmin=-1, vmax=1, cmap='RdBu_r'):
    client_n = similarities.shape[0]
    client_id = [i+1 for i in range(client_n)]
    ax = sns.heatmap(similarities, annot=True, vmin=vmin, vmax=vmax, cmap=cmap, xticklabels=client_id, yticklabels=client_id)
    plt.show()
    