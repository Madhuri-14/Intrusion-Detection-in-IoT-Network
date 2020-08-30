import pdb
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import math
#import matplotlib.pyplot as plt
#from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
#from sklearn.tree import export_graphviz
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import cross_val_score
import os
import subprocess
#from glob import glob
from time import time
from operator import itemgetter
from scipy.stats import randint
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=DeprecationWarning)

SVM_MODEL = "SVC"


# not being used currently
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


def read_records():
    NORMAL_TRAFFIC = 0
    MIRAI_ATTACK = 1
    MITM_ATTACK = 2
    SCAN_ATTACK = 3
    
    # Reading File
    df_benign_new = pd.read_csv("C:\\Master's\\Spring 2020\\Graduate Research Assistanship\\Dataset\\iot-network-intrusion-dataset-CSV-FORMAT\\benign\\benign-dec.csv", usecols = [10,13,28])
    df_benign = df_benign_new.head(n=29000)
    df_mirai_attack_new = pd.read_csv("C:\\Master's\\Spring 2020\\Graduate Research Assistanship\\Dataset\\iot-network-intrusion-dataset-CSV-FORMAT\\mirai\\mirai-ackflooding-1-dec.csv",usecols = [10,13,28]) 
    df_mirai_attack = df_mirai_attack_new.head(n=29000)
    df_mitm_attack_new = pd.read_csv ("C:\\Master's\\Spring 2020\\Graduate Research Assistanship\\Dataset\\iot-network-intrusion-dataset-CSV-FORMAT\\mitm\\mitm-arpspoofing-1-dec.csv",usecols = [10,13,28]) 
    df_mitm_attack = df_mitm_attack_new.head(n=29000)
    df_scan_attack_new = pd.read_csv("C:\\Master's\\Spring 2020\\Graduate Research Assistanship\\Dataset\\iot-network-intrusion-dataset-CSV-FORMAT\\scan\\scan-hostport-1-dec.csv",usecols = [10,13,28]) 
    df_scan_attack = df_scan_attack_new.head(n=29000)
    '''
    filenames = glob("C:\\Master's\\Spring 2020\\Graduate Research Assistanship\\Dataset\\iot-network-intrusion-dataset-CSV-FORMAT\\mirai\\mirai-ackflooding-1-dec.csv") 
    attack_list = list()
    for f in filenames:
        attack_list.append(pd.read_csv(f))
        df_mirai_attack = pd.concat(attack_list, ignore_index=True)
    
    filenames = glob("C:\\Master's\\Spring 2020\\Graduate Research Assistanship\\Dataset\\iot-network-intrusion-dataset-CSV-FORMAT\\mitm\\mitm-arpspoofing-1-dec.csv") 
    attack_list = list()
    for f in filenames:
        attack_list.append(pd.read_csv(f))
        df_mitm_attack = pd.concat(attack_list, ignore_index=True)
    
    filenames = glob("C:\\Master's\\Spring 2020\\Graduate Research Assistanship\\Dataset\\iot-network-intrusion-dataset-CSV-FORMAT\\scan\\scan-hostport-1-dec.csv") 
    attack_list = list()
    for f in filenames:
        attack_list.append(pd.read_csv(f))
        df_scan_attack = pd.concat(attack_list, ignore_index=True)

    '''
    # Create Label for classification
    df_benign['label'] = NORMAL_TRAFFIC
    df_mirai_attack['label'] = MIRAI_ATTACK
    df_mitm_attack['label'] = MITM_ATTACK
    df_scan_attack['label'] = SCAN_ATTACK
    

    df_all = pd.concat([df_benign, df_mirai_attack, df_mitm_attack, df_scan_attack], ignore_index=True)
    return df_all


def train_test_split( df_all):
    train_test_split_percentage = 80
    threshold = train_test_split_percentage / 100
    np.random.seed(5000)
    number_of_records = len(df_all)
    msk = np.random.rand(number_of_records) < threshold # splitting 80% training and remaining 20% as testing
    #msk array will have [True/False... number_of_records]
    train = df_all[msk]   #Taking true(msk) into training data
    test = df_all[~msk]   #taking false(msk) into testing data

    # There are sklearn functions which allows you to split data into train and test

    trainX = train.iloc[:,0:-1]
    trainY = train.iloc[:,-1]
    testX = test.iloc[:,0:-1]
    testY = test.iloc[:,-1]

    return trainX, trainY, testX, testY


def create_model(model):
    if model == SVM_MODEL:
        clf = SVC()
    return clf



def main():

    df_all = read_records()
    trainX, trainY, testX, testY = train_test_split(df_all)
    
    clf = create_model(SVM_MODEL)

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
    print ("Final Training RMSE: ", train_rmse)
    print("Training Accuracy Score:", train_scores)
    print ("Final Testing RMSE: ", test_rmse)
    print("Testing Accuracy score: ", test_scores)
  
    pass

main()

