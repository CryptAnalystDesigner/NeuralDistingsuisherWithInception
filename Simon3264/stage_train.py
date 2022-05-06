from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model,model_from_json
from tensorflow.keras.backend import concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, AveragePooling1D, Conv1D, MaxPooling1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.nn import dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# import keras
from pickle import dump
import tensorflow as tf
import simon as si
import numpy as np
import multiprocessing as mp
import tensorflow
import gc
from numba import cuda 
# import os


bs = 2000
# wdir = './freshly_trained_nets/'
wdir = './good_trained_nets/'
# 不断修改学习率

strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
print('Number of devices: %d' % strategy.num_replicas_in_sync) 
process_number = 60
def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)
def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)


def first_stage(n,num_rounds=12,pairs=8):

    
    test_n = int(n/10)
    

    # X, Y = si.make_train_data(n, nr=num_rounds-3, pairs=pairs,diff=(0x0440, 0x0100))
    # X_eval,Y_eval = si.make_train_data(test_n, nr=num_rounds-3, pairs=pairs,diff=(0x0440, 0x0100))
    
    # 生成训练数据
    # 使用的服务器核数
    
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(si.make_train_data, [(int(n/process_number),num_rounds-3,pairs,(0x0440, 0x0100),) for i in range(process_number)])

    X = accept_XY[0][0]
    Y = accept_XY[0][1]

    for i in range(process_number-1):
        X = np.concatenate((X,accept_XY[i+1][0]))
        Y = np.concatenate((Y,accept_XY[i+1][1]))

    with mp.Pool(process_number) as pool:
        accept_XY_eval = pool.starmap(si.make_train_data, [(int(test_n/process_number),num_rounds-3,pairs,(0x0440, 0x0100),) for i in range(process_number)])

    X_eval = accept_XY_eval[0][0]
    Y_eval = accept_XY_eval[0][1]
    
    for i in range(process_number-1):
        X_eval = np.concatenate((X_eval,accept_XY_eval[i+1][0]))
        Y_eval = np.concatenate((Y_eval,accept_XY_eval[i+1][1]))

    print("multiple processing end ......")

    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model(wdir+"simon_model_"+str(num_rounds-1)+"r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
        net_json = net.to_json()

        net_first = model_from_json(net_json)
        net_first.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        # net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_first.load_weights(wdir+"simon_model_"+str(num_rounds-1)+"r_depth5_num_epochs20_pairs"+str(pairs)+".h5")

    check = make_checkpoint(
        wdir+'first_best'+str(num_rounds)+"_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_first.fit(X, Y, epochs=20, batch_size=batch_size,
                  validation_data=(X_eval, Y_eval),callbacks=[lr,check])

    net_first.save(wdir+"net_first.h5")
    device = cuda.get_current_device()
    device.reset()
 

def second_stage(n,num_rounds=12, pairs=8):

    
    # n=10**8
    test_n = int(n/10)
    # X, Y = si.make_train_data(n, nr=num_rounds, pairs=pairs)
    # X_eval, Y_eval = si.make_train_data(test_n, nr=num_rounds, pairs=pairs)
    
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(si.make_train_data, [(int(n/process_number),num_rounds,pairs,) for i in range(process_number)])

    X = accept_XY[0][0]
    Y = accept_XY[0][1]
    
    for i in range(process_number-1):
        X = np.concatenate((X,accept_XY[i+1][0]))
        Y = np.concatenate((Y,accept_XY[i+1][1]))
 
    with mp.Pool(process_number) as pool:
        accept_XY_eval = pool.starmap(si.make_train_data, [(int(test_n/process_number),num_rounds,pairs,) for i in range(process_number)])

    X_eval = accept_XY_eval[0][0]
    Y_eval = accept_XY_eval[0][1]
    
    for i in range(process_number-1):
        X_eval = np.concatenate((X_eval,accept_XY_eval[i+1][0]))
        Y_eval = np.concatenate((Y_eval,accept_XY_eval[i+1][1]))

    print("multiple processing end ......")


    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model(wdir+"net_first.h5")
        net_json = net.to_json()

        net_second = model_from_json(net_json)
        net_second.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        # net_second.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_second.load_weights(wdir+"net_first.h5")
        
    
    check = make_checkpoint(
        wdir+'second_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_second.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check])

    net_second.save(wdir+"net_second.h5")
    device = cuda.get_current_device()
    device.reset()


def stage_train(n,num_rounds=12, pairs=8):

    
    # n=10**8
    test_n = int(n/10)

    # X, Y = si.make_train_data(n, nr=num_rounds, pairs=pairs)
    # X_eval, Y_eval = si.make_train_data(test_n, nr=num_rounds, pairs=pairs)
    
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(si.make_train_data, [(int(n/process_number),num_rounds,pairs,) for i in range(process_number)])

    X = accept_XY[0][0]
    Y = accept_XY[0][1]
    
    for i in range(process_number-1):
        X = np.concatenate((X,accept_XY[i+1][0]))
        Y = np.concatenate((Y,accept_XY[i+1][1]))
   

    
    with mp.Pool(process_number) as pool:
        accept_XY_eval = pool.starmap(si.make_train_data, [(int(test_n/process_number),num_rounds,pairs,) for i in range(process_number)])

    X_eval = accept_XY_eval[0][0]
    Y_eval = accept_XY_eval[0][1]
    
    for i in range(process_number-1):
        X_eval = np.concatenate((X_eval,accept_XY_eval[i+1][0]))
        Y_eval = np.concatenate((Y_eval,accept_XY_eval[i+1][1]))
    

    print("multiple processing end ......")


    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model(wdir+"net_second.h5")
        net_json = net.to_json()

        net_third = model_from_json(net_json)
        net_third.compile(optimizer=Adam(learning_rate = 10**-5), loss='mse', metrics=['acc'])
        # net_third.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_third.load_weights(wdir+"net_second.h5")

    check = make_checkpoint(
        wdir+'third_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_third.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check])

    net_third.save(wdir+"simon_model_"+str(num_rounds)+"r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
   


if __name__ == "__main__":

    first_stage( n=10**7,num_rounds=12,pairs=4)
    # second_stage(n=10**7,num_rounds=12,pairs=4)
    # stage_train( n=10**7,num_rounds=12,pairs=4)