# This file contains the model implementations to be imported
import numpy as np
from tensorflow import keras
from keras import models, layers, losses, Sequential
from keras import backend as K
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import time


def augmentation():
    """
    Data augmentation function - not currently in use

    Returns:
        Augmentation layer (add it after Input layer if desired)
    """
    augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.05, 0.05),
        layers.GaussianNoise(0.01),
        ])
    return augmentation


def AdaptiveMaxPooling1D(x, output_shape):
    """
    Implementation of Adaptive Max Pooling 1D procedure

    Adaptive Max Pooling is equivalent to Max Pooling, \
        but uses an adaptive set of parameters (kernel \
        and strides) in order to reach a desired fixed \
        output shape, regardless of input shape.
    """
    outputshape = output_shape # desired output shape

    dims = np.array(K.int_shape(x)[1:-1]) # get dimensions

    strides = np.floor(dims/outputshape).astype(np.int32)

    kernels = dims - ((outputshape-1) * strides)

    return layers.MaxPooling1D(pool_size=kernels, strides=strides)


def AdaptiveMaxPooling2D(x, output_shape):
    """
    Implementation of Adaptive Max Pooling 2D procedure

    Adaptive Max Pooling is equivalent to Max Pooling, \
        but uses an adaptive set of parameters (kernel \
        and strides) in order to reach a desired fixed \
        output shape, regardless of input shape.
    """
    outputshape = output_shape # desired output shape

    dims = np.array(K.int_shape(x)[1:-1]) # get dimensions

    strides = np.floor(dims/outputshape).astype(np.int32)

    kernels = dims - ((outputshape-1) * strides)

    return layers.MaxPooling2D(pool_size=kernels, strides=strides)


def MLP(input_shape, num_classes, neurons_first=1024, n_layers=5):
    """
    MLP model based on Zhao
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # Flatten input as it is all dense layers
    x = layers.Flatten()(input_)

    # 5 layers, each decaying by half in number of neurons
    for i in range(n_layers):
        x = layers.Dense(neurons_first / 2**i)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Output layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='MLP')

    # Compile model
    loss = losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics='accuracy')

    return model


def encoder_1D(input_shape, neurons=1024, n_layers=5):
    """
    Encoder model 1D based on Zhao

    This encoder is used for AE 1D, SAE 1D and DAE 1D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # Flatten input as it is all dense layers
    x = layers.Flatten()(input_)

    # 5 layers, each decaying by half in number of neurons
    for i in range(n_layers):
        x = layers.Dense(neurons / 2**i)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='encoder_1D')

    return model


def decoder_1D(input_shape, output_shape, neurons=1024, n_layers=5):
    """
    Decoder model 1D based on Zhao

    This decoder is used for AE 1D, SAE 1D and DAE 1D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # Flatten input as it is all dense layers
    x = layers.Flatten()(input_)

    # N layers, each doubling its number of neurons
    for i in reversed(range(n_layers-1)):
        x = layers.Dense(neurons / 2**i)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # And a last layer to adjust to output shape
    x = layers.Dense(np.prod(output_shape))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Reshape back to input shape if multiple dimension
    x = layers.Reshape((output_shape))(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='decoder_1D')

    return model


def encoder_2D(input_shape, kernel=(3,3)):
    """
    Encoder model 2D based on Zhao

    This encoder is used for AE 2D, SAE 2D and DAE 2D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # First conv block
    x = layers.Conv2D(filters=3, kernel_size=kernel, padding='same')(input_)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second conv block: strides=(2,2) to halve input shape
    x = layers.Conv2D(filters=3, kernel_size=kernel, padding='same',
                      strides=(2,2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Two more conv blocks, both with 32 filters and strides=(1,1)
    for _ in range(2):
        x = layers.Conv2D(filters=32, kernel_size=kernel,
                          padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Flatten input
    x = layers.Flatten()(x)

    # Then 2 fully connected layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='encoder_2D')

    return model


def decoder_2D(input_shape, kernel=(3,3), output_shape=None):
    """
    Decoder model 2D based on Zhao

    This decoder is used for AE 2D, SAE 2D and DAE 2D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # Opposite as encoder: starting with 2 fully connected layers
    x = layers.Dense(256, activation='relu')(input_)
    x = layers.Dense(8192, activation='relu')(x)

    # Reshape to (16, 16, 32)
    x = layers.Reshape((16,16,-1))(x)

    # 2 convs transpose with 32 filters and stride (1,1)
    for _ in range(2):
        x = layers.Conv2DTranspose(filters=32, kernel_size=kernel,
                                   padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # 1 conv transpose with 3 filters and stride (2,2) to double input shape
    x = layers.Conv2DTranspose(filters=3, kernel_size=kernel,
                               strides=(2,2), padding='same',
                               output_padding=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # We need to fit output shape to encoder's input shape with pad if needed
    size = K.int_shape(x)[1:-1]
    if output_shape and size != output_shape:
        # Calculate amount of pad needed
        pad = np.subtract(output_shape[:-1], size)
        # Rewrite pad in the form ((top,bottom), (left,right))
        pad = [(np.floor(item/2).astype(np.int16),
                np.ceil(item/2).astype(np.int16)) for item in pad]
        # Add padding layer
        x = layers.ZeroPadding2D(padding=pad)(x)

    # 1 more conv with 1 filter to convert back to input shape
    x = layers.Conv2DTranspose(filters=1, kernel_size=kernel,
                               padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='decoder_2D')

    return model


def AE_classifier_1D(input_shape, n_classes, loss='MSE', name='AE'):
    """
    Constructing AE 1D based on Zhao

    This AE is a concatenation of encoder 1D and decoder 1D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # Encoder imported from function
    encoder = encoder_1D(input_shape)(input_)

    # Decoder imported from function: input shape is output of encoder
    encoder_input = K.int_shape(input_)[1:]
    encoder_output = K.int_shape(encoder)[1:]
    decoder = decoder_1D(encoder_output, encoder_input)(encoder)

    # Building classifier as a separate exit path from encoder
    classif = layers.Activation('relu')(encoder)
    classif = layers.Dense(n_classes, activation='softmax')(classif)

    # Build model: starts on input_ and ends on decoder AND classif (2 outputs)
    model = models.Model(input_, [decoder, classif],
                         name=name + '_classifier_1D')

    # Compile model
    if 'KLD' in loss.upper() or 'KLP' in loss.upper():
        loss = losses.KLDivergence()
    elif 'MSE' in loss.upper():
        loss = losses.MeanSquaredError()

    loss2 = losses.CategoricalCrossentropy()

    # 1 loss per output (loss for encoder-decoder, loss2 for encoder-classif)
    model.compile(loss=[loss, loss2], optimizer='Adam')

    return model


def AE_classifier_2D(input_shape, n_classes, loss='MSE', name='AE'):
    """
    Constructing AE 2D based on Zhao

    This AE is a concatenation of encoder 2D and decoder 2D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # Encoder imported from function
    encoder = encoder_2D(input_shape)(input_)

    # Decoder imported from function: input shape is output of encoder
    encoder_input = K.int_shape(input_)[1:]
    encoder_output = K.int_shape(encoder)[1:]
    decoder = decoder_2D(encoder_output, output_shape=encoder_input)(encoder)

    # Building classifier as a separate exit path from encoder
    classif = layers.Activation('relu')(encoder)
    classif = layers.Dense(n_classes)(classif)

    # Build model: starts on input_ and ends on decoder AND classif (2 outputs)
    model = models.Model(input_, [decoder, classif],
                         name=name + '_classifier_2D')

    # Compile model
    if 'KLD' in loss.upper() or 'KLP' in loss.upper():
        loss = losses.KLDivergence()
    elif 'MSE' in loss.upper():
        loss = losses.MeanSquaredError()

    loss2 = losses.CategoricalCrossentropy()

    # 1 loss per output (loss for encoder-decoder, loss2 for encoder-classif)
    model.compile(loss=[loss, loss2], optimizer='Adam')

    return model


def SAE_classifier_1D(input_shape, n_classes, loss='KLD'):
    """
    Constructing SAE 1D based on Zhao

    AE, DAE and SAE all have the same structure

    SAE uses KLD loss instead of MSE
    """
    return AE_classifier_1D(input_shape, n_classes, loss='KLD', name='SAE')


def SAE_classifier_2D(input_shape, n_classes, loss='KLD'):
    """
    Constructing SAE 2D based on Zhao

    AE, DAE and SAE all have the same structure

    SAE uses KLD loss instead of MSE
    """
    return AE_classifier_2D(input_shape, n_classes, loss='KLD', name='SAE')


def DAE_classifier_1D(input_shape, n_classes, loss='MSE'):
    """
    Constructing AE 1D based on Zhao

    AE, DAE and SAE all have the same structure

    DAE basically adds noise to the input on preprocessing
    """
    return AE_classifier_1D(input_shape, n_classes, loss='MSE', name='DAE')


def DAE_classifier_2D(input_shape, n_classes, loss='MSE'):
    """
    Constructing AE 2D based on Zhao

    AE, DAE and SAE all have the same structure

    DAE basically adds noise to the input on preprocessing
    """
    return AE_classifier_2D(input_shape, n_classes, loss='MSE', name='DAE')


def conv_1D(input_shape, n_classes):
    """
    CNN 1D based on Zhao
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # 2 Conv Layers, each with BatchNorm layer
    x = layers.Conv1D(filters=16, kernel_size=15, padding='valid')(input_)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters=32, kernel_size=3, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply max pooling
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Again, 2 Conv Layers, each with BatchNorm layer
    x = layers.Conv1D(filters=64, kernel_size=3, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters=128, kernel_size=3, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply Adaptive Max Pooling
    x = AdaptiveMaxPooling1D(x, output_shape=4)(x)

    # Flatten, then 3 fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=256, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=n_classes, activation='softmax')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='conv_1D')

    # Compile model
    loss = losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics='accuracy')

    return model


def conv_2D(input_shape, n_classes):
    """
    CNN 2D based on Zhao
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # 2 Conv Layers, each with BatchNorm layer
    x = layers.Conv2D(filters=16, kernel_size=(3,3), padding='valid')(input_)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply max pooling
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    # Again, 2 Conv Layers, each with BatchNorm layer
    x = layers.Conv2D(filters=64, kernel_size=(3,3), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=128, kernel_size=(3,3), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply Adaptive Max Pooling
    x = AdaptiveMaxPooling2D(x, output_shape=4)(x)

    # Flatten, then 3 fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=1024, activation='relu')(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=n_classes, activation='softmax')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='conv_2D')

    # Compile model
    loss = losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics='accuracy')

    return model


def LeNet_1D(input_shape, n_classes):
    """
    LeNet model 1D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # 1 Conv Layer with BatchNorm layer
    x = layers.Conv1D(filters=6, kernel_size=5, padding='valid')(input_)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply max pooling
    #x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding='valid', input_shape=(input_shape[1], input_shape[2]))(x)

    # Again, 1 Conv Layer with BatchNorm layer
    x = layers.Conv1D(filters=16, kernel_size=5, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply Adaptive Max Pooling
    x = AdaptiveMaxPooling1D(x, output_shape=25)(x)

    # Flatten, then 3 fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=120, activation='relu')(x)
    x = layers.Dense(units=84, activation='relu')(x)
    x = layers.Dense(units=n_classes, activation='softmax')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='LeNet_1D')

    # Compile model
    loss = losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics='accuracy')

    return model


def LeNet_2D(input_shape, n_classes):
    """
    LeNet model 2D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # 1 Conv Layer with BatchNorm layer
    x = layers.Conv2D(filters=6, kernel_size=(5,5), padding='valid')(input_)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply max pooling
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    # Again, 1 Conv Layer with BatchNorm layer
    x = layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply Adaptive Max Pooling
    x = AdaptiveMaxPooling2D(x, output_shape=5)(x)

    # Flatten, then 3 fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=120, activation='relu')(x)
    x = layers.Dense(units=84, activation='relu')(x)
    x = layers.Dense(units=n_classes, activation='softmax')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='LeNet_2D')

    # Compile model
    loss = losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics='accuracy')

    return model


def AlexNet_1D(input_shape, n_classes):
    """
    AlexNet model 1D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # 2 Conv Layers with BatchNorm layer and Max Pooling each
    x = layers.Conv1D(filters=64, kernel_size=11, strides=4,
                      padding='valid')(input_)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=192, kernel_size=5, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2)(x)

    # Then 3 Conv Layers with BatchNorm layer each
    x = layers.Conv1D(filters=384, kernel_size=3, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same')(x)
    x = layers.Activation('relu')(x)

    # Apply Adaptive Max Pooling
    x = AdaptiveMaxPooling1D(x, output_shape=6)(x)

    # Flatten, then 3 fully connected layers, first 2 with dropout
    x = layers.Flatten()(x)
    x = layers.Dense(units=1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=n_classes, activation='softmax')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='AlexNet_1D')

    # Compile model
    loss = losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics='accuracy')

    return model


def AlexNet_2D(input_shape, n_classes):
    """
    AlexNet model 2D
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # 2 Conv Layers with BatchNorm layer and Max Pooling each
    x = layers.Conv2D(filters=64, kernel_size=(11,11), strides=(4,4),
                      padding='valid')(input_)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = layers.Conv2D(filters=192, kernel_size=(5,5), padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    # Then 3 Conv Layers with BatchNorm layer each
    x = layers.Conv2D(filters=384, kernel_size=(3,3), padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2),
                      padding='same')(x)
    x = layers.Activation('relu')(x)

    # Apply Adaptive Max Pooling
    x = AdaptiveMaxPooling2D(x, output_shape=6)(x)

    # Flatten, then 3 fully connected layers, first 2 with dropout
    x = layers.Flatten()(x)
    x = layers.Dense(units=4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=n_classes, activation='softmax')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='AlexNet_2D')

    # Compile model
    loss = losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics='accuracy')

    return model


def BiLSTM_1D(input_shape, n_classes):
    """
    BiLSTM model 1D based on Zhao
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # 2 Conv Layers with BatchNorm layer and Max Pooling each
    x = layers.Conv1D(filters=16, kernel_size=3, padding='same')(input_)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=32, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply Adaptive Max Pooling
    x = AdaptiveMaxPooling1D(x, output_shape=25)(x)

    # Permute (transpose/swap) first and second dimensions
    # x = layers.Permute((2,1))(x)

    # Then, apply bidirectional LSTM block
    x = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x)

    # Flatten, then 2 fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=256, activation='relu')(x)
    x = layers.Dense(units=n_classes, activation='softmax')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='BiLSTM_1D')

    # Compile model
    loss = losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics='accuracy')

    return model


def BiLSTM_2D(input_shape, n_classes):
    """
    BiLSTM model 2D based on Zhao
    """
    # Initiate input
    input_ = layers.Input(shape=input_shape)

    # 2 Conv Layers with BatchNorm layer and Max Pooling each
    x = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same')(input_)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Apply Adaptive Max Pooling
    x = AdaptiveMaxPooling2D(x, output_shape=5)(x)

    # Reshape
    x = layers.Reshape((-1, 32))(x)

    # Permute (transpose/swap) first and second dimensions
    # x = layers.Permute((2,1))(x)

    # Then, apply bidirectional LSTM block
    x = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x)

    # Flatten, then 2 fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=256, activation='relu')(x)
    x = layers.Dense(units=n_classes, activation='softmax')(x)

    # Build model: starts on input_ and ends on last x
    model = models.Model(input_, x, name='BiLSTM_2D')

    # Compile model
    loss = losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics='accuracy')

    return model

def nb(x_train,x_test,y_train,y_test):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report
    
    param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    naive_bayes = GaussianNB()
    print("Starting grid search...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    grid_search_nb = GridSearchCV(naive_bayes, param_grid_nb, cv=3, scoring='accuracy')
    y_train_cat = np.argmax(y_train, axis=1)
    grid_search_nb.fit(x_train, y_train_cat)
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Grid search time:", tempo_decorrido, " Seconds")

    print("Best parameters for naive bayes:",grid_search_nb.best_params_)
    print("Best score for naive bayes:",grid_search_nb.best_score_)

    print("Starting model...")

    naive = GaussianNB(var_smoothing= grid_search_nb.best_params_['var_smoothing'])
    print("Start training...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    naive.fit(x_train, y_train_cat)
    print("Training complete")
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Training time:", tempo_decorrido, " Seconds")
    print("Testing accuracy...")
    y_test_cat = np.argmax(y_test, axis=1)
    naive_predic = naive.predict(x_test)
    print('Acurácia:',accuracy_score(y_test_cat, naive_predic))
    print(classification_report(y_test_cat, naive_predic))

    cm_nb = confusion_matrix(y_test_cat, naive_predic)
    ax = sns.heatmap(cm_nb / np.sum(cm_nb), annot=True, fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix - NAIVE BAYES\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()

    return


def tree_decision(x_train,x_test,y_train,y_test):
    from sklearn.tree import DecisionTreeClassifier

    param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 40],
    'min_samples_split': [2, 5, 10],
    }

    dt = DecisionTreeClassifier(random_state=0)
    print("Starting grid search...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=None, scoring='accuracy')
    grid_search_dt = GridSearchCV(dt, param_grid = param_grid_dt, cv=None)
    grid_search_dt.fit(x_train, y_train)
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Grid search time:", tempo_decorrido, " Seconds")

    print("Best parameters for decision tree:", grid_search_dt.best_params_)
    print("Best score for decision tree:", grid_search_dt.best_score_)

    print("Starting model...")
    decision_tree = OneVsRestClassifier(DecisionTreeClassifier(criterion= grid_search_dt.best_params_['criterion'], max_depth= grid_search_dt.best_params_['max_depth'], min_samples_split=grid_search_dt.best_params_['min_samples_split']))
    print("Start training...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    print(x_train.shape,y_train.shape)
    decision_tree.fit(x_train, y_train)
    print("Training complete")
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Training time:", tempo_decorrido, " Seconds")
    print("Testing accuracy...")
    tree_predict = decision_tree.predict(x_test)
    print('Acurácia:',accuracy_score(y_test, tree_predict))
    print(classification_report(y_test, tree_predict))

    y_categorical = np.argmax(y_test, axis=1)
    predict_categorical = np.argmax(tree_predict, axis=1)

    cm_dt = confusion_matrix(y_categorical, predict_categorical)
    ax = sns.heatmap(cm_dt / np.sum(cm_dt), annot=True, fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix - Decision Tree\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()

    return

def rand_forest(x_train,x_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    }
    rf = RandomForestClassifier(random_state=42)
    print("Starting grid search...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    grid_search_rf = GridSearchCV(rf, param_grid_rf , cv=3, scoring='accuracy')
    grid_search_rf.fit(x_train, y_train)
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Grid search time:", tempo_decorrido, " Seconds")
    print("Best parameters for random forest:",grid_search_rf.best_params_)
    print("Best score for random forest:",grid_search_rf.best_score_)

    print("Starting model...")
    modelo_rf = RandomForestClassifier(criterion=grid_search_rf.best_params_['criterion'], max_depth= grid_search_rf.best_params_['max_depth'], n_estimators= grid_search_rf.best_params_['n_estimators'])
    print("Start training...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    modelo_rf.fit(x_train, y_train)
    print("Training complete")
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Training time:", tempo_decorrido, " Seconds")
    print("Testing accuracy...")

    randomF_predic = modelo_rf.predict(x_test)
    print('Acurácia:',accuracy_score(y_test, randomF_predic))
    print(classification_report(y_test, randomF_predic))

    y_categorical = np.argmax(y_test, axis=1)
    predict_categorical = np.argmax(randomF_predic, axis=1)

    cm_dt = confusion_matrix(y_categorical, predict_categorical)
    ax = sns.heatmap(cm_dt / np.sum(cm_dt), annot=True, fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix - Random Forest\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()

    return

def SupportVectorMachine(x_train,x_test,y_train,y_test):
    param_grid_svm = {
    'C': [1, 10, 1e2, 1e3, 1e4, 1e5],
    'kernel': ['linear', 'poly', 'rbf']
    }
    y_train_cat = np.argmax(y_train, axis=1)
    svm = SVC(random_state=42, max_iter=100)
    print("Starting grid search...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    grid_search_svm = GridSearchCV(svm, param_grid = param_grid_svm, cv=3)
    grid_search_svm.fit(x_train, y_train_cat)
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Grid search time:", tempo_decorrido, " Seconds")

    print("Best parameters for SVM:",grid_search_svm.best_params_)
    print("Best score for SVM:",grid_search_svm.best_score_)

    modelo_svm = SVC(kernel=grid_search_svm.best_params_['kernel'], C=grid_search_svm.best_params_['C'])
    print("Starting training...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    modelo_svm.fit(x_train, y_train_cat)
    print("Training complete")
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Training time:", tempo_decorrido, " Seconds")
    print("Testing accuracy...")

    svm_predict = modelo_svm.predict(x_test)

    y_test_cat = np.argmax(y_test, axis=1)
    print('Acurácia:',accuracy_score(y_test_cat, svm_predict))
    print(classification_report(y_test_cat, svm_predict))

    cm_svm = confusion_matrix(y_test_cat, svm_predict)
    ax = sns.heatmap(cm_svm/np.sum(cm_svm), annot=True, fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix - SVM\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()

    return

def multiLayerPerceptron(x_train,x_test,y_train,y_test):
    from sklearn.neural_network import MLPClassifier
    param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01],
    }
    mlp = MLPClassifier(random_state=0, max_iter=50)
    print("Starting grid search...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, cv=3, scoring='accuracy')
    grid_search_mlp.fit(x_train, y_train)
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Grid search time:", tempo_decorrido, " Seconds")

    print("Best parameters for MLP:",grid_search_mlp.best_params_)
    print("Best score for MLP:",grid_search_mlp.best_score_)
    print("Starting model...")
    # Marca o tempo inicial
    tempo_inicial = time.time()
    clf = MLPClassifier(activation=grid_search_mlp.best_params_['activation'], alpha=grid_search_mlp.best_params_['alpha'], hidden_layer_sizes=grid_search_mlp.best_params_['hidden_layer_sizes'] ,solver='adam')
    print("Start training...")
    clf.fit(x_train, y_train)
    
    print("Training complete")
    # Marca o tempo final
    tempo_final = time.time()
    tempo_decorrido = tempo_final - tempo_inicial
    print("Training time:", tempo_decorrido, " Seconds")
    print("Testing accuracy...")

    mlp_predict = clf.predict(x_test)
    print('Acurácia:',accuracy_score(y_test, mlp_predict))
    print(classification_report(y_test, mlp_predict))

    y_categorical = np.argmax(y_test, axis=1)
    predict_categorical = np.argmax(mlp_predict, axis=1)

    cm_mlp = confusion_matrix(y_categorical, predict_categorical)
    ax = sns.heatmap(cm_mlp/np.sum(cm_mlp), annot=True, fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix - MLP\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()

    return

def cnn(x_train,x_test,y_train,y_test):
    from keras.optimizers import Adam
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.layers import LeakyReLU
    from sklearn import metrics
    from sklearn.model_selection import train_test_split

    x_train, X_val, y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    model = Sequential() 

    model.add(Conv2D(filters = 16, kernel_size = 3,  padding='same', activation = 'relu', input_shape = (x_train.shape[1] , x_train.shape[2] , x_train.shape[3])))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Conv2D(filters = 32, kernel_size = 3,  padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Conv2D(filters = 64, kernel_size = 3,  padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Dropout(0.3))
    model.add(Flatten()) 
    model.add(Dense(200, activation=LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.5)) 

    model.add(Dense(50, activation=LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.3)) 

    model.add(Dense(y_train.shape[1] , activation = 'softmax'))

    #model.summary()

    INIT_LR = 1e-2
    EPOCHS = 100
    BS = 32

    opt = Adam(lr = INIT_LR , decay = INIT_LR / EPOCHS)
    model.compile(loss = "categorical_crossentropy" , optimizer = opt , metrics = ["accuracy"])
    H = model.fit(x_train, y_train , steps_per_epoch = len(x_train) // BS , validation_data = (X_val , Y_val) , validation_steps = len(X_val) // BS , epochs = EPOCHS)

    y_pred = model.predict(x_test)
    y_pred_new = np.argmax(y_pred , axis = 1) # To get the index (The class numer) of the predicted class
    y_test_new = np.argmax(y_test , axis = 1)

    Confusion_Mtrx = metrics.confusion_matrix(y_test_new , y_pred_new)

    print('Acurácia:',accuracy_score(y_test_new, y_pred_new))
    print(classification_report(y_test_new, y_pred_new))

    cm_mlp = confusion_matrix(y_test_new, y_pred_new)
    ax = sns.heatmap(cm_mlp/np.sum(cm_mlp), annot=True, fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix - CNN\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()


def rnn(x_train,x_test,y_train,y_test, n_classes):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, TimeDistributed, BatchNormalization
    from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
    from keras.layers import LeakyReLU
    # from keras.utils import np_utils
    from tensorflow.keras.utils import plot_model
    import itertools

    model = Sequential([
        SimpleRNN(units=50, activation='tanh', return_sequences=True, input_shape=x_train.shape[1:]),
        LSTM(units=50, return_sequences=True, input_shape=x_train.shape[1:]),
        GRU(units=50, return_sequences=True, input_shape=x_train.shape[1:]),
        #BatchNormalization(),
        Flatten(), 
        Dense(200, activation=LeakyReLU(alpha=0.3)),
        Dropout(0.5),
        Dense(50, activation=LeakyReLU(alpha=0.3)),
        Dropout(0.3),
        Dense(units=6, activation = 'softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()

    # train model
    model_history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.1)

    #
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('GRU model loss - clean')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # test
    test_loss = model.evaluate(x_test[:50], y_test[:50], batch_size=256)
    test_RMSE = np.sqrt(test_loss)
    print(test_RMSE)

    # plot predicted RUL
    y_pred = model.predict(x_test)
    y_pred_new = np.argmax(y_pred , axis = 1) # To get the index (The class numer) of the predicted class
    y_test_new = np.argmax(y_test , axis = 1)

    cm_rnn = confusion_matrix(y_test_new, y_pred_new)
    ax = sns.heatmap(cm_rnn/np.sum(cm_rnn), annot=True, fmt='.2%', cmap='Blues')
    ax.set_title('Confusion Matrix - GRU\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()

def vae_mlp(x_train, x_test):
    import keras
    import tensorflow.compat.v1 as tf
    from keras.models import Sequential
    from keras.losses import binary_crossentropy, categorical_crossentropy, mse
    from keras.layers import Lambda, Input, Dense, Dropout
    from keras.models import Model
    from keras.utils import plot_model
    from keras import backend as K
    from sklearn.preprocessing import StandardScaler # to use Standard Scaler, replace for: MinMaxScaler for StandardScaler

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import math
    #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
    #sc =  StandardScaler() 
    original_dim = x_train.shape[1]
    #x_train = sc.fit_transform(x_train) # Always fit the scaler to the training dataset. For the test dataset, just transform it.
    #x_test = sc.transform(x_test)

    def sampling(args):
    
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean = 0, stddev = 0.01)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    input_shape = (original_dim, )
    batch_size = 32
    latent_dim = 2
    epochs = 10
  # build encoder model

    inputs = Input(shape=input_shape, name='encoder_input')
    h1 = Dense(512, activation='relu')(inputs)
    h2 = Dense(256, activation='relu')(h1)
    h3 = Dense(128, activation='relu')(h2)
    z_mean = Dense(latent_dim, name='z_mean')(h3)
    z_log_var = Dense(latent_dim, name='z_log_var')(h3)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    h4 = Dense(128, activation='relu')(latent_inputs)
    h5 = Dense(256, activation='relu')(h4)
    h6 = Dense(512, activation='relu')(h5)
    outputs = Dense(original_dim, activation='sigmoid')(h6)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = mse(inputs, outputs)
    print(inputs)
    print(outputs)
    reconstruction_loss *= original_dim

    #Now we define the KL dicvergence:
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    vae.compile(optimizer = 'adam', metrics = ['accuracy'])

    vae.summary()
    VAE_train = vae.fit(x_train, epochs = epochs, batch_size = batch_size, validation_split = 0.1)
    result = vae.evaluate(x_test)   
    print(result)
    x_test_encoded = encoder.predict(x_test, batch_size = batch_size)

    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[2][:, 0], x_test_encoded[2][:, 1])
    plt.colorbar()
    plt.show()
    
    return

def vae_co(X_train,X_test):
    import tensorflow as tf
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model
    from keras.losses import mse, binary_crossentropy
    from keras import backend as K
    from sklearn.cluster import KMeans
    from keras.optimizers import Adam
    import hdbscan
    import matplotlib.pyplot as plt

    def VAE_summary (Model):
        plt.plot(Model.history['loss'])
        plt.plot(Model.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

    def sampling(args):
        
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim), mean = 0, stddev = 0.01)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
    # Parâmetros do VAE
    original_dim = X_train.shape[1]
    latent_dim = 8
    batch_size = 32
    epochs = 100

    inputs = Input(shape=original_dim, name='encoder_input')
    h1 = Dense(100, activation='relu')(inputs)
    h2 = Dense(50, activation='relu')(h1)
    h3 = Dense(25, activation='relu')(h2)
    z_mean = Dense(latent_dim, name='z_mean')(h3)
    z_log_var = Dense(latent_dim, name='z_log_var')(h3)

    z = Lambda(sampling, name='z')([z_mean, z_log_var])

    # encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    h4 = Dense(25, activation='relu')(latent_inputs)
    h5 = Dense(50, activation='relu')(h4)
    h6 = Dense(100, activation='relu')(h5)
    outputs = Dense(original_dim, activation='sigmoid')(h6)

    # decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = mse(inputs, outputs)
    print(inputs)
    print(outputs)
    reconstruction_loss *= original_dim

    #KL divergence:
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    learning_rate = 0.001
    optimizer = Adam(lr=learning_rate)
    vae.compile(optimizer = optimizer)

    vae.summary()
    VAE_train = vae.fit(X_train, epochs = epochs, batch_size = batch_size, validation_split = 0.1)
    #vae.fit(X_train, epochs=epochs, batch_size=batch_size)

    # Obter representações latentes
    encoder = Model(inputs, z_mean)
    latent_representation = encoder.predict(X_test)


    # Inicialize o modelo HDBSCAN com os parâmetros desejados
    min_cluster_size = 20
    min_samples = None  # Pode ser ajustado conforme necessário
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Ajuste o modelo aos dados de representação latente
    cluster_labels = clusterer.fit_predict(latent_representation)

    VAE_summary(VAE_train)

    # Plotar os dados do espaço latente com cores diferentes para cada cluster
    plt.figure(figsize=(8, 6))
    for i in range(cluster_labels.max() + 1):
        plt.scatter(latent_representation[cluster_labels == i, 0],
                    latent_representation[cluster_labels == i, 1],
                    label=f'Cluster {i}', alpha=0.5)
    plt.title('Espaço Latente com Clusters (HDBSCAN)')
    plt.xlabel('Latent Feature 1')
    plt.ylabel('Latent Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


    return

