# from dataset_process.dataset_process import MyDataset
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE


def plot_with_labels(lowDWeights, labels, kinds, file_name):
    """
    Plot a 2D clustering result and annotate each point with its label.
    
    Args:
        lowDWeights: The 2D embedding of the data (after dimensionality reduction).
        labels: Labels corresponding to each data point.
        kinds: Number of categories/classes.
        file_name: Name of the file to save the figure.
    """
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    
    # Draw each point with its corresponding label
    for x, y, s in zip(X, Y, labels):
        # To ensure color distinction, split the 0–255 color range into `kinds` parts
        c = cm.rainbow(int(255 / kinds * s))
        plt.text(x, y, s, backgroundcolor=c, fontsize=6)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())

    plt.title('Clustering Step-Wise after Embedding')
    plt.rcParams['figure.figsize'] = (10.0, 10.0)

    if not os.path.exists(f'gather_figure/{file_name.split(" ")[0]}'):
        os.makedirs(f'gather_figure/{file_name.split(" ")[0]}')

    plt.savefig(f'gather_figure/{file_name.split(" ")[0]}/{file_name}.jpg', dpi=600)
    plt.close()


def plot_only(lowDWeights, labels, index, file_name):
    """
    Plot a 2D clustering result with custom coloring rules for labels.
    
    Args:
        lowDWeights: 2D embedding of the data (after dimensionality reduction).
        labels: Labels corresponding to each data point.
        index: Index used to distinguish output files (avoid overwriting).
        file_name: File name and clustering method used.
    """
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]

    # Custom color rules for different regions
    for x, y, s in zip(X, Y, labels):
        position = 255
        if x < -850:
            position = 255
        elif 0.5 * x - 225 < y:
            position = 0
        elif x < 1500:
            position = 50
        else:
            position = 100

        c = cm.rainbow(position)
        plt.text(x, y, s, backgroundcolor=c, fontsize=6)

    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.rcParams['figure.figsize'] = (10.0, 10.0)

    if not os.path.exists(f'gather_figure/{file_name.split(" ")[0]}'):
        os.makedirs(f'gather_figure/{file_name.split(" ")[0]}')

    plt.savefig(f'gather_figure/{file_name.split(" ")[0]}/{file_name} {index}.jpg', dpi=600)
    plt.close()


def gather_by_tsne(X: np.ndarray,
                   Y: np.ndarray,
                   index: int,
                   file_name: str):
    """
    Perform dimensionality reduction with t-SNE and plot clustering.
    
    Args:
        X: Data to cluster (high-dimensional).
        Y: Labels corresponding to X.
        index: File index to distinguish saved figures.
        file_name: Output file name.
    """
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=4000)
    low_dim_embs = tsne.fit_transform(X[:, :])
    labels = Y[:]
    plot_only(low_dim_embs, labels, index, file_name)


def gather_all_by_tsne(X: np.ndarray,
                       Y: np.ndarray,
                       kinds: int,
                       file_name: str):
    """
    Perform dimensionality reduction with t-SNE and plot clustering (with multiple categories).
    
    Args:
        X: Data to cluster (2D data).
        Y: Labels corresponding to X.
        kinds: Number of categories/classes.
        file_name: Output file name.
    """
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=4000)
    low_dim_embs = tsne.fit_transform(X[:, :])
    labels = Y[:]
    plot_with_labels(low_dim_embs, labels, kinds, file_name)


if __name__ == '__main__':
    path = 'E:\\PyCharmWorkSpace\\mtsdata\\ECG\\ECG.mat'  # length=100, input=152, channel=2, output=2

    # dataset = MyDataset(path, 'train')
    # X = dataset.train_dataset
    # X = torch.mean(X, dim=1).numpy()
    # Y = dataset.train_label.numpy()

    # Example usage (provide index and file_name)
    # gather_by_tsne(X, Y, index=0, file_name="ECG_TSNE")
