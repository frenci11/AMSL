import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as scio
import random
from random import choice
from scipy.signal import savgol_filter
from sklearn import preprocessing

global folder_name, a, dim_DA
a = 3601
dim_DA = 500

folder_name = "FordA"

def check():
    # Define folder and URL
    zip_url = "https://www.timeseriesclassification.com/aeon-toolkit/FordA.zip"
    zip_file = "FordA.zip"

    # Check if the folder exists
    if not os.path.exists(folder_name):
        print(f"Folder '{folder_name}' not found. Downloading and extracting...")

        # Download the ZIP file
        urllib.request.urlretrieve(zip_url, zip_file)
        print("Download complete.")

        # Extract the ZIP file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(folder_name)
        print(f"Extracted contents to '{folder_name}'.")

        # Clean up: delete the downloaded zip file
        os.remove(zip_file)
        print("Temporary zip file removed.")
    else:
        print(f"Folder '{folder_name}' already exists. No action taken.")



def inter_data(hr, window=11):
    N = window
    time3 = savgol_filter(hr, window_length=N, polyorder=2)
    return time3

def noised(signal):
    SNR = 5
    noise = np.random.randn(signal.shape[0])
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(signal) ** 2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise

def negated(signal):
    return signal * -1

def opposite_time(signal):
    return signal[::-1]

def permuted(signal):
    listA = [0,1,2,3,4]
    sliceSize, rem = divmod(signal.shape[0], len(listA))
    assert not rem
    random.shuffle(listA)
    sig = signal[listA[0]*sliceSize:listA[0]*sliceSize+sliceSize]
    for i in range(1,len(listA)):
        sig = np.hstack((sig,signal[listA[i]*sliceSize:listA[i]*sliceSize+sliceSize]))
    return sig

def scale(signal):
    sc = [0.5, 2, 1.5, 0.8]
    s = choice(sc)
    return signal * s

def time_warp(signal):
    return inter_data(signal,11)

def regular_mm(data):
    preShape = data.shape
    data = data.reshape(-1, 1)# ()
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = data.reshape((preShape[0], preShape[1]))
    return data

def augment(dataX):
    # Augment the dataset
    data_no = np.zeros((dataX.shape[0],dataX.shape[1]))
    data_ne = np.zeros((dataX.shape[0],dataX.shape[1]))
    data_op = np.zeros((dataX.shape[0], dataX.shape[1]))
    data_pe = np.zeros((dataX.shape[0], dataX.shape[1]))
    data_sc = np.zeros((dataX.shape[0], dataX.shape[1]))
    data_ti = np.zeros((dataX.shape[0], dataX.shape[1]))

    for i in range(dataX.shape[0]):
        data_no[i] = noised(dataX[i].copy())
        data_ne[i] = negated(dataX[i].copy())
        data_op[i] = opposite_time(dataX[i].copy())
        data_pe[i] = permuted(dataX[i].copy())
        data_sc[i] = scale(dataX[i].copy())
        data_ti[i] = time_warp(dataX[i].copy())

    #####################Normalization###########################
    data_raw = regular_mm(dataX)
    data_no = regular_mm(data_no)
    data_ne = regular_mm(data_ne)
    data_op= regular_mm(data_op)
    data_pe = regular_mm(data_pe)
    data_sc = regular_mm(data_sc)
    data_ti = regular_mm(data_ti)

    data_raw = np.reshape(data_raw, data_raw.shape + (1,))
    data_no = np.reshape(data_no, data_no.shape + (1,))
    data_ne = np.reshape(data_ne, data_ne.shape + (1,))
    data_op = np.reshape(data_op, data_op.shape + (1,))
    data_pe = np.reshape(data_pe, data_pe.shape + (1,))
    data_sc = np.reshape(data_sc, data_sc.shape + (1,))
    data_ti = np.reshape(data_ti, data_ti.shape + (1,))

    return data_raw,data_no,data_ne,data_op,data_pe,data_sc,data_ti


if __name__ == "__main__":
    check()

    train_dataset_filename = os.path.join(folder_name,'FordA_TRAIN.txt')
    test_dataset_filename = os.path.join(folder_name,'FordA_TEST.txt')

    train_dataset = np.loadtxt(train_dataset_filename)
    test_dataset = np.loadtxt(test_dataset_filename)  

    X_train = train_dataset[:,1:]
    Y_train = train_dataset[:,0]

    X_train_normal = X_train[Y_train == 1]
    X_train_abnormal = X_train[Y_train == -1]

    X_test = test_dataset[:,1:]
    Y_test = test_dataset[:,0]

    print(f'Normal train dataset shape: {X_train_normal.shape}')
    print(f'Abnormal train dataset shape: {X_train_abnormal.shape}')

    print(f'Test dataset shape: {test_dataset.shape}')
    
    data_raw,data_no,data_ne,data_op,data_pe,data_sc,data_ti = augment(np.vstack((X_train_normal,X_train_abnormal, X_test)))

    X_train_normal_normalized = np.array((
        data_raw[:X_train_normal.shape[0],:],
        data_no[:X_train_normal.shape[0],:],
        data_ne[:X_train_normal.shape[0],:],
        data_op[:X_train_normal.shape[0],:],
        data_pe[:X_train_normal.shape[0],:],
        data_sc[:X_train_normal.shape[0],:],
        data_ti[:X_train_normal.shape[0],:]
    ))

    X_train_normal_normalized = np.transpose(X_train_normal_normalized, (1,0,2,3))
    # adds an axis since out dataset is 1D but the net accepts 2D timeseries as input
    X_train_normal_normalized = X_train_normal_normalized[:, :, :, :, np.newaxis]

    X_train_abnormal_normalized = np.array((
        data_raw[X_train_normal.shape[0]:X_train_normal.shape[0]+X_train_abnormal.shape[0],:],
        data_no[X_train_normal.shape[0]:X_train_normal.shape[0]+X_train_abnormal.shape[0],:],
        data_ne[X_train_normal.shape[0]:X_train_normal.shape[0]+X_train_abnormal.shape[0],:],
        data_op[X_train_normal.shape[0]:X_train_normal.shape[0]+X_train_abnormal.shape[0],:],
        data_pe[X_train_normal.shape[0]:X_train_normal.shape[0]+X_train_abnormal.shape[0],:],
        data_sc[X_train_normal.shape[0]:X_train_normal.shape[0]+X_train_abnormal.shape[0],:],
        data_ti[X_train_normal.shape[0]:X_train_normal.shape[0]+X_train_abnormal.shape[0],:]
    ))

    X_train_abnormal_normalized = np.transpose(X_train_abnormal_normalized, (1,0,2,3))
    # adds an axis since out dataset is 1D but the net accepts 2D timeseries as input
    X_train_abnormal_normalized = X_train_abnormal_normalized[:, :, :, :, np.newaxis]

    X_test_normalized = np.array((
        data_raw[X_train_normal.shape[0]+X_train_abnormal.shape[0]:,:],
        data_no[X_train_normal.shape[0]+X_train_abnormal.shape[0]:,:],
        data_ne[X_train_normal.shape[0]+X_train_abnormal.shape[0]:,:],
        data_op[X_train_normal.shape[0]+X_train_abnormal.shape[0]:,:],
        data_pe[X_train_normal.shape[0]+X_train_abnormal.shape[0]:,:],
        data_sc[X_train_normal.shape[0]+X_train_abnormal.shape[0]:,:],
        data_ti[X_train_normal.shape[0]+X_train_abnormal.shape[0]:,:]
    ))

    X_test_normalized = np.transpose(X_test_normalized, (1,0,2,3))
    # adds an axis since out dataset is 1D but the net accepts 2D timeseries as input
    X_test_normalized = X_test_normalized[:, :, :, :, np.newaxis]
    
    Path(os.path.join(folder_name, 'normalized')).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(folder_name, 'normalized', 'X_train_normal.npy'), X_train_normal_normalized)
    np.save(os.path.join(folder_name, 'normalized', 'X_train_abnormal.npy'), X_train_abnormal_normalized)

    np.save(os.path.join(folder_name, 'normalized', 'X_test.npy'), X_test_normalized)
    np.save(os.path.join(folder_name, 'normalized', 'Y_test.npy'), Y_test)



