#Universidade Federal de Pernambuco
#1EE - Machine Learning
#Alunos: Gabriela Farias, Gustavo Camargo 

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from functions import *
from models import *


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("Nenhuma GPU detectada. Certifique-se de que o TensorFlow-GPU está instalado corretamente.")
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU {physical_devices[0]} configurada para uso.")

x_train, x_test, y_train, y_test, n_classes, df = load_3w("../data/",preprocessing=None, scaler=StandardScaler())
#x_train, x_test, y_train, y_test, n_classes, df = load_3w("data/",preprocessing=None)
#print(x_train.shape, y_train.shape)

print(y_test)
print("Classes únicas no treino:", np.unique(y_train))
print("Classes únicas no teste:", np.unique(y_test))

#x_train, x_test, y_train, y_validacao = train_test_split(xx, yy, test_size=0.2, random_state=42)

print(x_train.shape)

#dados = dados.drop('T-JUS-CKGL', axis=1)
#dados = dados.drop('timestamp', axis=1)
#dados = dados.dropna() #drop de todas linhas com NaN

#contagem_classes = dados['class'].value_counts()

#print(contagem_classes)

#rotulos = analise(dados)

x_train = x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0],-1)

#x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])



print(x_train.shape, x_test.shape)






