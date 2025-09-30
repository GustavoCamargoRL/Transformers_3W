import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch


def heatMap_all(score_input: torch.Tensor,  # 2D tensor
                score_channel: torch.Tensor,  # 2D tensor
                x: torch.Tensor,  # 2D tensor
                save_root: str,
                file_name: str,
                accuracy: str,
                index: int) -> None:
    score_channel = score_channel.detach().numpy()
    score_input = score_input.detach().numpy()
    draw_data = x.detach().numpy()

    # Distance function for DTW (here simply the absolute difference)
    euclidean_norm = lambda x, y: np.abs(x - y)

    # Matrix to record DTW distance between channels
    matrix_00 = np.ones((draw_data.shape[1], draw_data.shape[1]))
    # Matrix to record L2 distance between inputs
    matrix_11 = np.ones((draw_data.shape[0], draw_data.shape[0]))

    # Compute L2 distances between input sequences
    for i in range(draw_data.shape[0]):
        for j in range(draw_data.shape[0]):
            x = draw_data[i, :]
            y = draw_data[j, :]
            matrix_11[i, j] = np.sqrt(np.sum((x - y) ** 2))

    # Compute DTW distances between channels
    draw_data = draw_data.transpose(-1, -2)
    for i in range(draw_data.shape[0]):
        for j in range(draw_data.shape[0]):
            x = draw_data[i, :]
            y = draw_data[j, :]
            d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
            matrix_00[i, j] = d

    plt.rcParams['figure.figsize'] = (10.0, 8.0)

    plt.subplot(221)
    sns.heatmap(score_channel, cmap="YlGnBu", vmin=0)
    # plt.title('channel-wise attention')

    plt.subplot(222)
    sns.heatmap(matrix_00, cmap="YlGnBu", vmin=0)
    # plt.title('channel-wise DTW')

    plt.subplot(223)
    sns.heatmap(score_input, cmap="YlGnBu", vmin=0)
    # plt.title('step-wise attention')

    plt.subplot(224)
    sns.heatmap(matrix_11, cmap="YlGnBu", vmin=0)
    # plt.title('step-wise L2 distance')

    # Save results
    if not os.path.exists(f'{save_root}/{file_name}'):
        os.makedirs(f'{save_root}/{file_name}')
    plt.savefig(f'{save_root}/{file_name}/{file_name} accuracy={accuracy} {index}.jpg', dpi=400)

    plt.close()


if __name__ == '__main__':
    matrix = torch.Tensor(range(24)).reshape(2, 3, 4)
    print(matrix.shape)
    file_name = 'lall'
    epoch = 1

    data_channel = matrix.detach()
    data_input = matrix.detach()

    plt.subplot(2, 2, 1)
    sns.heatmap(data_channel[0].data.cpu().numpy())
    plt.title("1")

    plt.subplot(2, 2, 2)
    sns.heatmap(data_input[0].data.cpu().numpy())
    plt.title("2")

    plt.subplot(2, 2, 3)
    sns.heatmap(data_input[0].data.cpu().numpy())
    plt.title("3")

    plt.subplot(2, 2, 4)
    sns.heatmap(data_input[0].data.cpu().numpy())
    plt.title("4")

    plt.suptitle("JapaneseVowels Attention Heat Map", fontsize='x-large', fontweight='bold')
    # plt.savefig('result_figure/JapaneseVowels Attention Heat Map.png')
    plt.show()
