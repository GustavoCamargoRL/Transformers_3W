import torch
print('Current PyTorch version:', torch.__version__)
from utils.random_seed import setup_seed
from torch.utils.data import DataLoader
from dataset_process.dataset_process import MyDataset
from utils.heatMap import heatMap_all
from utils.TSNE import gather_by_tsne
from utils.TSNE import gather_all_by_tsne
import numpy as np
from utils.colorful_line import draw_colorful_line

setup_seed(30)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Dataset dimensions overview
# ArabicDigits length=6600  input=93 channel=13 output=10
# AUSLAN length=1140  input=136 channel=22 output=95
# CharacterTrajectories
# CMUsubject16 length=29,29  input=580 channel=62 output=2
# ECG  length=100  input=152 channel=2 output=2
# JapaneseVowels length=270  input=29 channel=12 output=9
# Libras length=180  input=45 channel=2 output=15
# UWave length=4278  input=315 channel=3 output=8
# KickvsPunch length=10  input=841 channel=62 output=2
# NetFlow length=803  input=997 channel=4 output=only 1 and 13 (dataset processing code needs modification)
# Wafer length=803  input=997 channel=4

# Select the model to run
save_model_path = 'saved_model/ECG 91.0 batch=2.pkl'
file_name = save_model_path.split('/')[-1].split(' ')[0]
path = f'E:\PyCharmWorkSpace\\dataset\\MTS_dataset\\{file_name}\\{file_name}.mat'  # Construct dataset path

# Preparation for naming HeatMap
ACCURACY = save_model_path.split('/')[-1].split(' ')[1]  # Accuracy of the model used
BATCH_SIZE = int(save_model_path[save_model_path.find('=')+1:save_model_path.rfind('.')])  # Batch size used by the model
heatMap_or_not = False  # Whether to plot the HeatMap of the Score matrix
gather_or_not = False  # Whether to plot clustering diagrams of a single sample on steps and channels
gather_all_or_not = True  # Whether to plot clustering diagram of all samples after feature extraction

# Load model
net = torch.load(save_model_path, map_location=torch.device('cpu'))  # map_location sets device; original pkl may have been trained on GPU in Colab
# Load test dataset
test_dataset = MyDataset(path, 'test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'Number of samples with maximum step length: {len(test_dataset.max_length_sample_inTest)}')
if len(test_dataset.max_length_sample_inTest) == 0:
    gather_or_not = False
    heatMap_or_not = False
    print('No samples with maximum step length found in test set. Cannot draw meaningful heatmap or gather plots. Try another dataset.')

correct = 0
total = 0
with torch.no_grad():
    all_sample_X = []
    all_sample_Y = []
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre, encoding, score_input, score_channel, gather_input, gather_channel, gate = net(x.to(DEVICE), 'test')

        all_sample_X.append(encoding)
        all_sample_Y.append(y)
        if heatMap_or_not:
            for index, sample in enumerate(test_dataset.max_length_sample_inTest):
                if sample.numpy().tolist() in x.numpy().tolist():
                    target_index = x.numpy().tolist().index(sample.numpy().tolist())
                    print('Drawing heatmap...')
                    heatMap_all(score_input[target_index], score_channel[target_index], sample, 'heatmap_figure_in_test', file_name, ACCURACY, index)
                    print('Heatmap completed!')
        if gather_or_not:
            for index, sample in enumerate(test_dataset.max_length_sample_inTest):
                if sample.numpy().tolist() in x.numpy().tolist():
                    target_index = x.numpy().tolist().index(sample.numpy().tolist())
                    print('Drawing gather plot...')
                    gather_by_tsne(gather_input[target_index].numpy(), np.arange(gather_input[target_index].shape[0]), index, file_name+' input_gather')
                    gather_by_tsne(gather_channel[target_index].numpy(), np.arange(gather_channel[target_index].shape[0]), index, file_name+' channel_gather')
                    print('Gather plot completed!')
                    draw_data = x[target_index].transpose(-1, -2)[0].numpy()
                    draw_colorful_line(draw_data)
                    gather_or_not = False

        _, label_index = torch.max(y_pre.data, dim=-1)
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()

    if gather_all_or_not:
        all_sample_X = torch.cat(all_sample_X, dim=0).numpy()
        all_sample_Y = torch.cat(all_sample_Y, dim=0).numpy()
        print('Drawing gather plot for all samples...')
        gather_all_by_tsne(all_sample_X, all_sample_Y, test_dataset.output_len, file_name+' all_sample_gather')
        print('Gather plot for all samples completed!')

    print(f'Accuracy: %.2f %%' % (100 * correct / total))
