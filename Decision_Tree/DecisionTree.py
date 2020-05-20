
import pandas as pd
import numpy as np
from sklearn import tree
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
#import pydotplus
#import Image
#from sklearn.externals.six import StringIO
import os
import subprocess
import sys
from time import time
from operator import itemgetter
from scipy.stats import randint
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def learning_curve(train_sizes, train_scores, test_scores):
    plt.title("Learning Curve")
    plt.ylim(0.990, 1.0)
    plt.xlabel("Decision Tree Depth")
    plt.ylabel("Accuracy Score")
    plt.grid()
    plt.plot(train_sizes, train_scores, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g",
             label="Testing score")

    plt.legend(loc="best")
    plt.show()
    pass


def dataset_training():
    benign_file =  sys.argv[1]
    attack_file = sys.argv[2]
    NORMAL_TRAFFIC = 0
    ATTACK = 1
    # Reading File
    df_benign = pd.read_csv(benign_file)
    df_attack = pd.read_csv(attack_file)

    # Create Label for classification
    df_benign['label'] = NORMAL_TRAFFIC
    df_attack['label'] = ATTACK

    df_all = pd.concat([df_benign, df_attack], ignore_index=True)

    np.random.seed(5000)

    msk = np.random.rand(len(df_all)) < 0.8 # splitting 80% training and remaining 20% as testing
    train = df_all[msk]
    test = df_all[~msk]
    trainX = train.iloc[:,0:-1]
    trainY = train.iloc[:,-1]
    testX = test.iloc[:,0:-1]
    testY = test.iloc[:,-1]

    clf = tree.DecisionTreeClassifier(max_depth = 5, criterion='entropy')

    clf = clf.fit(trainX, trainY)
    result = clf.predict(trainX)
    predY = pd.Series(result)
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    train_rmse = rmse
    train_scores = accuracy_score(trainY, predY, normalize=True, sample_weight=None)

    result = clf.predict(testX)
    predY = pd.Series(result)
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    test_rmse = rmse
    test_scores = accuracy_score(testY, predY, normalize=True, sample_weight=None)
    print ("Final Training RMSE: ", train_rmse, train_scores)
    print ("Final Testing RMSE: ", test_rmse, test_scores)

    train_scores = []
    test_scores = []
    train_rmse = []
    test_rmse = []
    dataset_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1]
    #dataset_fractions = [0.2, 0.4, 0.6, 0.8, 1]
    for i in dataset_fractions:
        np.random.seed(5001)
        msk = np.random.rand(len(trainX)) < i
        temp_trainX = trainX[msk]
        temp_trainY = trainY[msk]
        clf = tree.DecisionTreeClassifier(max_depth = 5,criterion='entropy')
        clf = clf.fit(temp_trainX, temp_trainY)
        result = clf.predict(temp_trainX)
        predY = pd.Series(result)
        rmse = math.sqrt(((temp_trainY - predY) ** 2).sum()/trainY.shape[0])
        train_rmse = np.append(train_rmse,rmse)

        train_scores = np.append(train_scores, accuracy_score(temp_trainY, predY, normalize=True, sample_weight=None))

        result = clf.predict(testX)
        predY = pd.Series(result)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        test_rmse = np.append(test_rmse,rmse)

        test_scores = np.append(test_scores, accuracy_score(testY, predY, normalize=True, sample_weight=None))


    train_sizes = [ x * trainX.shape[0] for x in dataset_fractions ]
    learning_curve(train_sizes, train_scores, test_scores)
    #rmse_curve(train_sizes, train_rmse, test_rmse)
    print("Final Training RMSE: ", train_rmse[-1],train_scores[-1])
    print("Final Testing RMSE: ", test_rmse[-1],test_scores[-1])
    pass


def DTDepth_training():
    benign_file =  sys.argv[1]
    attack_file = sys.argv[2]
    NORMAL_TRAFFIC = 0
    ATTACK = 1
    # Reading File
    df_benign = pd.read_csv(benign_file)
    df_attack = pd.read_csv(attack_file)

    # Create Label for classification
    df_benign['label'] = NORMAL_TRAFFIC
    df_attack['label'] = ATTACK

    df_all = pd.concat([df_benign, df_attack], ignore_index=True)

    np.random.seed(5000)

    msk = np.random.rand(len(df_all)) < 0.8 # splitting 80% training and remaining 20% as testing
    train = df_all[msk]
    test = df_all[~msk]
    trainX = train.iloc[:,0:-1]
    trainY = train.iloc[:,-1]
    testX = test.iloc[:,0:-1]
    testY = test.iloc[:,-1]
    train_scores = []
    test_scores = []
    train_rmse = []
    test_rmse = []
    depths = [3,5,7,9,11]
    for i in depths:
        np.random.seed(5001)
        msk = np.random.rand(len(trainX)) < 0.6
        temp_trainX = trainX[msk]
        temp_trainY = trainY[msk]
        clf = tree.DecisionTreeClassifier(max_depth = i,criterion='entropy')
        clf = clf.fit(temp_trainX, temp_trainY)
        result = clf.predict(temp_trainX)
        predY = pd.Series(result)
        rmse = math.sqrt(((temp_trainY - predY) ** 2).sum()/trainY.shape[0])
        train_rmse = np.append(train_rmse,rmse)

        train_scores = np.append(train_scores, accuracy_score(temp_trainY, predY, normalize=True, sample_weight=None))

        result = clf.predict(testX)
        predY = pd.Series(result)
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        test_rmse = np.append(test_rmse,rmse)

        test_scores = np.append(test_scores, accuracy_score(testY, predY, normalize=True, sample_weight=None))


    train_sizes = depths
    learning_curve(train_sizes, train_scores, test_scores)
    print("Final Training RMSE: ", train_rmse[-1],train_scores[-1])
    print("Final Testing RMSE: ", test_rmse[-1],test_scores[-1])
    pass

dataset_training()
#DTDepth_training()
