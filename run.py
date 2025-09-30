import torch
from torch.utils.data import DataLoader
# from dataset_process.dataset_process import MyDataset
import torch.optim as optim
from time import time
from tqdm import tqdm
import os

from module.transformer import Transformer
from module.loss import Myloss
from utils.random_seed import setup_seed
from utils.visualization import result_visualization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from functions import *
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        # Converte one-hot para índice de classe
        if len(y.shape) > 1 and y.shape[1] > 1:
            self.y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)
            self.output_len = y.shape[1]  # Número de classes pelo tamanho do vetor one-hot
        else:
            self.y = torch.tensor(y, dtype=torch.long)
            self.output_len = int(self.y.max().item()) + 1
        self.train_len = len(self.x)
        self.test_len = len(self.x)
        self.input_len = self.x.shape[1] if self.x.ndim > 1 else 1
        self.channel_len = self.x.shape[2] if self.x.ndim > 2 else 1

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from mytest.gather.main import draw

setup_seed(30)  # Set the random seed
reslut_figure_path = 'result_figure'  # Path to save result images

# Dataset path selection
# path = '..\WalkvsRun\WalkvsRun.mat'
path = '..\AUSLAN\dataset_3W.mat'

test_interval = 5  # Test interval unit: epoch
draw_key = 1  # Images will only be saved when greater than or equal to draw_key.
file_name = path.split('\\')[-1][0:path.split('\\')[-1].index('.')]  # Get the file name

# Hyperparameter settings
EPOCH = 50
BATCH_SIZE = 64
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Select device: CPU or GPU
print(f'use device: {DEVICE}')

d_model = 512
d_hidden = 1024
q = 8  # Number of attention heads for the "query" dimension in multi-head attention
v = 8  # Number of attention heads for the "value" dimension in multi-head attention
h = 8  # Number of attention heads for the "key" dimension in multi-head attention
N = 8  # Number of encoder layers in the Transformer
dropout = 0.2
pe = True  # In the dual tower settings: score=pe and score=channel, default is pe.
mask = True  # In the dual tower configuration: score=input, default has no mask.
# Optimizer selection
optimizer_name = 'Adagrad'

dataset_path = r"D:\Profissional\DATASETS\3W_novo"
x_train, x_test, y_train, y_test, n_classes = load_3w_novo(dataset_path, window_length=16, preprocessing=None, scaler=StandardScaler())


# train_dataset = MyDataset(path, 'train')
# test_dataset = MyDataset(path, 'test')
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_dataset = NumpyDataset(x_train, y_train)
test_dataset = NumpyDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes, counts = np.unique(train_dataset.y.numpy(), return_counts=True)
print("Class distribution in train set:")
for c, n in zip(classes, counts):
    print(f"Class {c}: {n} instances")

# Para o conjunto de teste
classes, counts = np.unique(test_dataset.y.numpy(), return_counts=True)
print("\nClass distribution in test set:")
for c, n in zip(classes, counts):
    print(f"Class {c}: {n} instances")

DATA_LEN = train_dataset.train_len  # Number of training samples
d_input = train_dataset.input_len  # Number of timesteps
d_channel = train_dataset.channel_len  # Time series dimensions
d_output = train_dataset.output_len  # Number of classes

# Show dimensions
print('data structure: [lines, timesteps, features]')
print(f'train data size: [{DATA_LEN, d_input, d_channel}]')
print(f'test data size: [{train_dataset.test_len, d_input, d_channel}]')
print(f'Number of classes: {d_output}')

# Create Transformer model
net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
# Create loss function (CrossEntropyLoss used here)
loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

# To record accuracy changes
correct_on_train = []
correct_on_test = []
# To record loss changes
loss_list = []
time_cost = 0


def plot_confusion_matrix(dataloader, class_names=None):
    net.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            all_preds.extend(label_index.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()


# Test function
def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)

# Training function
def train():
    net.train()
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    begin = time()
    for index in range(EPOCH):
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')

            loss = loss_function(y_pre, y.to(DEVICE))

            print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()

        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f'Current max accuracy\tTest set:{max(correct_on_test)}%\t Train set:{max(correct_on_train)}%')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')

        pbar.update()

    os.makedirs("saved_model", exist_ok=True)
    dest_path = f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl'
    src_path = f'saved_model/{file_name} batch={BATCH_SIZE}.pkl'
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(src_path, dest_path)

    end = time()
    time_cost = round((end - begin) / 60, 2)

    # Result plots
    result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
                         test_interval=test_interval,
                         d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
                         time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
                         file_name=file_name,
                         optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)

    plot_confusion_matrix(test_dataloader, class_names=[str(i) for i in range(d_output)])


if __name__ == '__main__':
    train()
