# This file contains functions to be imported by your runs
import numpy as np
import pandas as pd
from glob import glob
from scipy.io import loadmat
from scipy.fft import fft
from scipy.signal import stft
from pywt import cwt
from sklearn.model_selection import train_test_split
from re import search
from ntpath import basename
import matplotlib.pyplot as plt
import seaborn as sns


def array_to_oh(array__):
    """
    Converts array to one-hot encoding
    """
    array_ = array__.astype(np.int16)
    onehot = np.zeros((array_.size, array_.max()+1))
    onehot[np.arange(array_.size), array_] = 1
    return onehot


def windowing_vectorized(array, window_length, step: int=None,
                         *, start=0, end: int=None,
                         axis=0):
    '''
    Windowing function > converts data points into sliding windows

    Args:
        data: array. Time points will be shifted to axis 0.
        window_length: length of the desired window. Defaults to 30.
        step: if specified, controls the starting point of the next \
            window. If None, step = window_length. Defaults to None.
        start: controls the starting point of the first window. \
            Defaults to 0.
        end: controls the ending point of the last window, if given. \
            May be truncated if (end - start) / step is not an \
            integer. Defaults to None.
        axis: axis of the time points. Defaults to 0.

    Returns:
        windowed data as array, with shape (windows, window_length, ...)
    '''
    if axis != 0:
        array = np.moveaxis(array, axis, 0)

    # Obtain the last possible point used by the window
    if (not end) or (end > len(array)):
        max_point = len(array) - start - window_length + 1
    else:
        max_point = end - start - window_length + 1

    # If step is not passed, use window_length: no points shared/skipped
    if not step:
        step = window_length

    sub_windows = (
        start +
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(window_length), 0) +
        np.expand_dims(np.arange(max_point, step=step), 1)
    )

    return array[sub_windows]


def perform_fft(input, axis=1):
    """
    Apply FFT transform to input data
    """
    return np.abs(fft(input, axis=axis))


def perform_stft(input, nperseg=64, axis=1):
    """
    Apply STFT transform to input data

    output's dimension is (floor[nperseg//2] + 1, \
                           ceil[len(input)//nperseg] * 2 + 1)

    For Zhao's work, where len(input)=1024 and nperseg=64, \
        the output shape is (33, 33)
    """
    _, _, Z = stft(input, nperseg=nperseg, axis=axis)
    return np.moveaxis(np.abs(Z),-1,axis+1)


def perform_cwt(input, length=20, axis=1):
    """
    Apply CWT transform to input data

    output's dimension is (len(input), length)

    For Zhao's work, where len(input)=100 for CWT and length=100, \
        the output shape is (100, 100)
    """
    # Set desired scales length
    if not length:
        length = input.shape[1]
    cwt_, _ = cwt(input, np.arange(1,length+1), 'mexh', axis=axis)
    return np.moveaxis(cwt_,0,axis+1)


def perform_image(input, axis=1):
    """
    Reshapes input data into two dimensions ("image")
    """
    # Move desired axis to last, if not already, to perform reshape correctly
    if axis != -1 and axis != (len(input.shape) - 1):
        input = np.moveaxis(input, axis, -1)

    # Write output shape: dimensions except last, then sqrt of last twice
    outputshape = (
        *input.shape[:-1],
        np.sqrt(input.shape[-1]).astype(np.int16),
        np.sqrt(input.shape[-1]).astype(np.int16),
    )

    # Reshape input into desired outputshape
    input = np.reshape(input, outputshape)

    # If axis was moved, revert the process (moveaxis twice due to two dims)
    if axis != -1 and (len(input.shape) - 1):
        input = np.moveaxis(input, -1, axis)
        input = np.moveaxis(input, -1, axis)

    return input


def organize_train_test(cases, window_length, preprocessing=None,
                        test_size=0.2, scaler=None, dataframes=None):
    """
    Function that applies preprocessing, splits train/test and scales
    """
    # Number of classes
    n_classes = len(cases)

    # Now open each case, create label and apply all processing
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i, case in enumerate(cases):
        # Apply windowing function (slicing function)
        x = windowing_vectorized(case, window_length=window_length)
        #print(len(x[0]))
        #print(i)
        # Create label
        y = np.ones(len(x)) * i
        #print(y)

        # Apply preprocessing
        if preprocessing:
            x = preprocessing(x)

        # Split train test
        train_test_x_y = train_test_split(x, y, test_size=test_size)

        # Append according to type
        x_train.append(train_test_x_y[0])
        x_test.append(train_test_x_y[1])
        y_train.append(train_test_x_y[2])
        y_test.append(train_test_x_y[3])

    # Shift from list to a concatenated array for each
    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Scale x data
    if scaler:
        x_train = scaler.fit_transform(x_train.reshape((
            x_train.shape[0], -1
        ))).reshape(x_train.shape)

        x_test = scaler.transform(x_test.reshape((
            x_test.shape[0], -1
        ))).reshape(x_test.shape)

    # Apply one-hot encoding to y
    y_train = array_to_oh(y_train)
    y_test = array_to_oh(y_test)

    # Shuffle x and y (for train/test) in unison
    p_train = np.random.permutation(len(x_train))
    p_test = np.random.permutation(len(x_test))

    # Return shuffled
    return x_train[p_train], x_test[p_test], \
           y_train[p_train], y_test[p_test], \
           n_classes, dataframes


def load_cwru(path_to_folder, window_length=1024, preprocessing=None,
              test_size=0.2, scaler=None, load='0'):
    """
    Load CWRU dataset, using each file as a class
    """
    # Getting data files for the load case only: RPM label
    dataset_files = glob(f'{path_to_folder}/**/*_{load}.mat', recursive=True)

    # Cases list created (healthy must be first element for its label to be 0)
    cases = []

    for file in dataset_files:
        # Load matlab file
        loaded = loadmat(file)

        # Access the specific matlab key to get the array data of Drive End
        key = [item for item in loaded.keys() if '_DE_time' in item]
        loaded = loaded[key[0]].reshape((-1))

        # healthy file will be inserted in the first position! (label=0)
        if f'normal_{load}' in file:
            cases.insert(0,loaded)
        else:
            cases.append(loaded)

    return organize_train_test(cases, window_length,
                               preprocessing, test_size, scaler, dataframes=None)


def load_mfpt(path_to_folder, window_length=1024, preprocessing=None,
              test_size=0.2, scaler=None):
    """
    Load MFPT dataset, using each file as a class, except \
        baseline and first three outer race faults
    """
    # Getting data files
    dataset_files = glob(f'{path_to_folder}/**/*.mat', recursive=True)

    # Folder to ignore mat files from
    exclude = ['5 - Analyses', '6 - Real World Examples']

    # Those cases have multiple files for the same class
    exceptions = {
        'baseline_[0-9].mat': [],
        'OuterRaceFault_[0-9].mat': [],
    }
    # The other cases refer to an unique file per class
    cases = []

    # Open and load all files
    for file in dataset_files:

        # Ignore files in the exclude list
        if any(element in file for element in exclude):
            continue

        loaded = loadmat(file)

        flag = False
        for case in exceptions:
            # Check if 'case' from exceptions is in file
            # Using search instead of 'if case in file' can use [0-9]
            if search(case, file):
                # This case has a problem, the bearing is in axis [0][0][1]
                if case == 'baseline_[0-9].mat':
                    exceptions[case].append(loaded['bearing'][0][0][1][:,0])
                else:
                    exceptions[case].append(loaded['bearing'][0][0][2][:,0])
                flag = True
                break
        if not flag:
            cases.append(loaded['bearing'][0][0][2][:,0])

    # From list to array
    baseline = np.array(exceptions['baseline_[0-9].mat'])
    outer = np.array(exceptions['OuterRaceFault_[0-9].mat'])

    # Take the mean for those cases with more than 1 file per class
    baseline = np.mean(baseline, axis=0)
    outer = np.mean(outer, axis=0)

    # Include all in same list
    # Healthy must be the first element for its label to be 0
    cases.insert(0, baseline)
    cases.insert(1, outer)

    return organize_train_test(cases, window_length,
                               preprocessing, test_size, scaler, dataframes=None)


def load_jnu(path_to_folder, window_length=1024, preprocessing=None,
             test_size=0.2, scaler=None):
    """
    Load JNU dataset, using each file as a class
    """
    # Getting data files
    dataset_files = glob(f'{path_to_folder}/**/*.csv', recursive=True)

    # Cases list created (healthy must be first element for its label to be 0)
    cases = []

    # Load files
    for file in dataset_files:
        # Some files have tab as separator, some have comma, so handle both
        loaded = pd.read_csv(file, header=None)
        loaded = np.array(loaded).reshape((-1,))

        # If the file is healthy, append to first position ("insert")
        if 'n600' in file.lower():
            cases.insert(0,loaded)
        # Else, append to last position
        else:
            cases.append(loaded)

    return organize_train_test(cases, window_length,
                               preprocessing, test_size, scaler, dataframes=None)


def load_xjtu(path_to_folder, window_length=1024, preprocessing=None,
              test_size=0.2, scaler=None):
    """
    Load XJTU dataset, using each bearing folder as a class
    """
    # Getting data files
    dataset_files = glob(f'{path_to_folder}/**/*.csv', recursive=True)

    # Cases used
    cases_dict = {
        f'Bearing{i}_{j}': []
        for i in range(1,4) for j in range(1,6)
    }

    # Get number of files for each case (folder)
    n_files_folder = {
        f'Bearing{i}_{j}': len(glob(f'{path_to_folder}/*/Bearing{i}_{j}/*'))
        for i in range(1,4) for j in range(1,6)
    }

    # Load files
    for file in dataset_files:
        # Add data to the specific case, and ignore if not in cases
        for case in cases_dict:
            if case in file:
                # Include only the 20 last csvs of each folder
                if int(basename(file)[:-4]) < n_files_folder[case] - 20:
                    continue

                # try:
                loaded = pd.read_csv(file, header=0)

                # Two columns of vib. signals: horizontal and vertical
                fh = np.array(loaded['Horizontal_vibration_signals'])
                fv = np.array(loaded['Vertical_vibration_signals'])

                f = np.concatenate((fh[..., np.newaxis],
                                    fv[..., np.newaxis]), axis=1)

                # cases_dict[case].append(f) # All data concatenated
                cases_dict[case].append(fh) # Horizontal vibration only
                # cases_dict[case].append(fv) # Vertical vibration only

                # except Exception:
                #     continue

    # Some cases have more than one file, so concatenate all into classes
    cases = []

    for case in cases_dict:
        try:
            data = np.concatenate(cases_dict[case], axis=0)
            cases.append(data)
        except ValueError:
            continue

    return organize_train_test(cases, window_length,
                               preprocessing, test_size, scaler, dataframes=None)


def load_seu(path_to_folder, window_length=1024, preprocessing=None,
             test_size=0.2, scaler=None):
    """
    Load SEU dataset, using each file as a class
    """
    # Getting data files
    dataset_files = glob(f'{path_to_folder}/**/*.csv', recursive=True)

    # Cases list created (healthy must be first element for its label to be 0)
    cases = []

    # Load files
    for file in dataset_files:
        # Some files have tab as separator, some have comma, so handle both
        loaded = pd.read_csv(file, skiprows=17, sep='[,\\t]', header=None,
                             names=list(range(8)), engine='python')

        # Zhao only uses second column of vibration data
        loaded = loaded.iloc[:,1]
        loaded = np.array(loaded)

        # If the file is healthy, append to first position ("insert")
        if 'health_' in file.lower():
            cases.insert(0,loaded)
        # Else, append to last position
        else:
            cases.append(loaded)

    return organize_train_test(cases, window_length,
                               preprocessing, test_size, scaler, dataframes=None)


def load_uoc(path_to_folder, window_length=1024, preprocessing=None,
             test_size=0.2, scaler=None):
    """
    Load UOC dataset \
        For the files, a 'class' changes evert 104 points in y axis
    """
    # Getting data files
    dataset_files = glob(f'{path_to_folder}/**/*.mat', recursive=True)

    # Cases list created (healthy must be first element for its label to be 0)
    cases = []

    for file in dataset_files:
        # Load matlab file
        loaded = loadmat(file)

        # Access the specific matlab key to get the array data of TimeDomain
        key = 'AccTimeDomain'
        try:
            loaded = loaded[key]
        except KeyError:
            continue

        # In this dataset, a 'class' changes every 104 points in y axis
        n_classes = len(loaded[0]) // 104

        for i in range(n_classes):
            start = i * 104
            end = (i+1) * 104
            class_element = loaded[:, start:end].reshape((-1))
            cases.append(class_element)

    return organize_train_test(cases, window_length,
                               preprocessing, test_size, scaler, dataframes=None)


def load_pu(path_to_folder, window_length=1024, preprocessing=None,
            test_size=0.2, scaler=None):
    """
    Load PU dataset, using each subdataset as a class
    """
    # Getting data files
    dataset_files = glob(f'{path_to_folder}/**/*.mat', recursive=True)

    # Subdatasets used by Zhao
    include = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24',
               'KB27', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']

    # Specific condition used by Zhao
    conditions = ['N15_M07_F10']

    # Cases dict created with empty lists
    cases_dict = {element: [] for element in include}

    for file in dataset_files:
        # Ignore conditions not in the conditions list
        if not any(element in file for element in conditions):
            continue

        # Add data to the specific 'include' case, and ignore if not in include
        for element in include:
            if element in file:
                # Load matlab file
                loaded = loadmat(file)

                # Access the specific matlab key to get the array data
                # key = list(loaded.keys())[-1]
                key = [item for item in loaded.keys() if '__' not in item][0]

                loaded = loaded[key]

                # Extract vibration data and reshape to vector
                vib_data = loaded[0][0][2][0][6][2]
                vib_data = np.reshape(vib_data, (-1,))

                cases_dict[element].append(vib_data)

    # Some cases have more than one file, so concatenate all into classes
    cases = []

    for case in cases_dict:
        try:
            data = np.concatenate(cases_dict[case], axis=0)
            cases.append(data)
        except ValueError:
            continue

    return organize_train_test(cases, window_length,
                               preprocessing, test_size, scaler, dataframes=None)


def load_3w(path_to_folder, window_length=64, preprocessing=None,
            test_size=0.2, scaler=None):
    """
    Load 3W dataset, using each folder as a class
    """
    # Getting data files
    dataset_files = glob(f'{path_to_folder}/**/*.csv', recursive=True)


    # Cases considered
    #cases = [f'\\3W\\{i}\\' for i in range(9)] # All cases
    cases = [f'\\{i}\\' for i in range(9)] # All cases
    # Excluding HAND DRAWN and SIMULATED files if desired
    exclude = ['DRAWN', 'SIMULATED']

    # Cases dict created with empty lists
    cases_dict = {element: [] for element in cases}
    dataframes = []
    for file in dataset_files:
        # Ignore cases in the exclude list
        #if any(element in file for element in exclude):
        #    continue

        # Add this file's data to the specific case
        for case in cases:
            if (case in file):
                # Load file
                dataframe = pd.read_csv(file)
                loaded = pd.read_csv(file, sep=',', header=0)
                dataframes.append(dataframe)

                # Drop columns 0 (timestamp), 7 (full nan) and 8 (class)
                loaded.drop(loaded.columns[[0,1,7,8,9]], axis=1, inplace=True)
                #loaded.drop(loaded.columns[[0,7]], axis=1, inplace=True)

                # Remove rows with NaNs
                loaded.dropna(inplace=True)

                # Convert to array
                loaded = np.array(loaded)

                cases_dict[case].append(loaded)
        

    # Some cases have more than one file, so concatenate all into classes
    cases = []
    dados = pd.concat(dataframes, ignore_index=True)

    print(dados.head())

    # Convert from dictionary to list, which is the format of other load funcs
    for case in cases_dict:
        try:
            data = np.concatenate(cases_dict[case], axis=0)

            # Append only if not empty
            if len(data) != 0:
                # Append to first position if healthy, append to end otherwise
                if case == '\\0\\':
                    cases.insert(0,data)
                else:
                    cases.append(data)
        except ValueError:
            continue
    

    return organize_train_test(cases, window_length,
                               preprocessing, test_size, scaler,dados)


def load_3w_v2(path_to_folder, window_length=64, preprocessing=None,
            test_size=0.2, scaler=None, single_class: int=None):
    """
    Load 3W dataset, using each folder as a class
    """
    # Getting data files
    dataset_files = glob(f'{path_to_folder}/**/*.csv', recursive=True)

    # Cases considered
    if single_class:
        cases = [f'\\{single_class}\\'] # Using single case
    else:
        cases = [f'\\{i}\\' for i in range(9)] # Using all cases

    # Excluding HAND DRAWN and SIMULATED files if desired
    exclude = ['DRAWN', 'SIMULATED']

    # Create empty dict to fill each label in
    cases_dict = {}

    for file in dataset_files:
        # Ignore cases not in used cases list
        if not any(element in file for element in cases):
            continue

        # Ignore cases in the exclude list
        if any(element in file for element in exclude):
            continue

        # Load file
        loaded = pd.read_csv(file, sep=',', header=0)

        # Drop columns 0 (timestamp) and 7 (full nan)
        loaded.drop(loaded.columns[[0,1,7,8,9]], axis=1, inplace=True)

        # Remove rows with NaNs
        loaded.dropna(inplace=True)

        # Convert to array
        loaded = np.array(loaded)

        # Split into x (data) and y (class)
        xx = loaded[...,:-1]
        yy = loaded[...,-1]

        # Get unique classes
        unique_classes = np.unique(yy)

        # Get data based on their class: each class is a case!
        for uni in unique_classes:
            # Get rows of that specific class only
            condition = np.where(yy==uni)
            data = xx[condition]

            # Check if this label already exists and create if not
            if uni not in cases_dict:
                cases_dict[uni] = []

            # Add data to cases dict
            cases_dict[uni].append(data)

    # Some cases have more than one file, so concatenate all into classes
    cases = []

    # Convert from dictionary to list, which is the format of other load funcs
    for case in cases_dict:
        try:
            data = np.concatenate(cases_dict[case], axis=0)

            # Append only if not empty
            if len(data) != 0:
                # Append to first position if healthy, append to end otherwise
                if case == '0.0':
                    cases.insert(0,data)
                else:
                    cases.append(data)
        except ValueError:
            continue

    return organize_train_test(cases, window_length,
                               preprocessing, test_size, scaler, dataframes=None)


def run_model(x_train, x_test, y_train, y_test,
              n_classes, scaler, model,
              batch_size=8, n_epochs=100, val_split=0.1,
              add_channel=False, sens=False):
    # Scale x data
    if scaler:
        x_train = scaler.fit_transform(x_train.reshape((
            x_train.shape[0], -1
        ))).reshape(x_train.shape)
        x_test = scaler.transform(x_test.reshape((
            x_test.shape[0], -1
        ))).reshape(x_test.shape)

    # Add channel axis to x data if needed
    if add_channel:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    # Load model
    model = model(x_train.shape[1:], n_classes)

    # If model is Denoising AE, add some noise to input
    if 'DAE_classifier' in model.name:
        x_train = x_train + np.random.normal(scale=0.01, size=x_train.shape)

    # If Model is 'AE', it has two outputs instead of one
    if 'AE_classifier' in model.name:
        y_train = [x_train, y_train]

    # Fit model
    verbose = 0 if sens else 2
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs,
              validation_split=val_split, verbose=verbose)

    # Get accuracy
    if 'AE_classifier' in model.name:
        out = model.predict(x_test, batch_size=2*batch_size,
                            verbose=verbose)
        acc = np.sum(
            np.argmax(out[1], axis=1) == np.argmax(y_test, axis=1)
            ) / len(y_test)
    else:
        out = model.evaluate(x_test, y_test, batch_size=2*batch_size,
                             verbose=verbose)
        acc = out[1]

    return acc

def analise(dados):
    from collections import Counter
    contagem_classes = dados['class'].value_counts().to_dict()

    contagem_classes[2.0] += contagem_classes.pop(102.0, 0)
    contagem_classes[5.0] += contagem_classes.pop(105.0, 0)

    rotulos = np.concatenate([[classe] * contagem for classe, contagem in contagem_classes.items()])
    rotulos = list(rotulos)
    contagem_rotulos = Counter(rotulos)
    for rotulo, quantidade in contagem_rotulos.items():
        print(f"Rótulo {rotulo}: {quantidade} vezes")

    dadosI = dados.drop(['class'], axis=1)
    estatisticas = dadosI.describe()
    print(estatisticas)
    variaveis = ['P-PDG', 'P-TPT', 'P-TPT',  'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'QGL']
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
    fig.suptitle('Gráfico de Dispersao')
    for i, variavel in enumerate(variaveis):
        linha = i // 4
        coluna = i % 4
        dados.boxplot(column=variavel, ax=axes[linha, coluna])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    matriz_correlacao = dadosI.corr()

    sns.heatmap(matriz_correlacao, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt=".2f")
    plt.show()

    return rotulos
