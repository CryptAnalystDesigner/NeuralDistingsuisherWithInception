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
import speck as sp
import numpy as np
import tensorflow
import gc
# import os


bs = 2000
wdir = './freshly_trained_nets/'
# wdir = './good_trained_nets/'
# 不断修改学习率

def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)
def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)


def first_stage(n,num_rounds=9,pairs=8):

    
    test_n = int(n/4)
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6"])
    
    print('Number of devices: %d' % strategy.num_replicas_in_sync) 
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model("model_7r_depth5_num_epochs20_pairs2.h5")
        net_json = net.to_json()

        net_first = model_from_json(net_json)
        # net_first.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_first.load_weights("model_7r_depth5_num_epochs20_pairs2.h5")

    X, Y = sp.make_train_data(n, nr=num_rounds-3, pairs=pairs,diff=(0x8000, 0x840a))
    X_eval,Y_eval = sp.make_train_data(test_n, nr=num_rounds-3, pairs=pairs,diff=(0x8000, 0x840a))
    
    # X, Y = sp.make_train_data(n, nr=num_rounds-2, pairs=pairs,diff=(0x8100,0x8102))
    # X_eval,Y_eval = sp.make_train_data(test_n, nr=num_rounds-2, pairs=pairs,diff=(0x8100,0x8102))

    check = make_checkpoint(
        wdir+'first_best'+str(num_rounds)+"_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_first.fit(X, Y, epochs=20, batch_size=batch_size,
                  validation_data=(X_eval, Y_eval),callbacks=[lr,check])

    net_first.save("net_first.h5")


def second_stage(n,num_rounds=9, pairs=8):

    
    # n=10**8
    test_n = int(n/4)
    X, Y = sp.make_train_data(n, nr=num_rounds, pairs=pairs)
    X_eval, Y_eval = sp.make_train_data(test_n, nr=num_rounds, pairs=pairs)
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync) 
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model("net_first.h5")
        net_json = net.to_json()

        net_second = model_from_json(net_json)
        net_second.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        # net_second.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_second.load_weights("net_first.h5")
        
    
    check = make_checkpoint(
        wdir+'second_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_second.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check])

    net_second.save("net_second.h5")


def stage_train(n,num_rounds=9, pairs=8):

    
    # n=10**8
    test_n = int(n/4)

    X, Y = sp.make_train_data(n, nr=num_rounds, pairs=pairs)
    X_eval, Y_eval = sp.make_train_data(test_n, nr=num_rounds, pairs=pairs)
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync) 
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model("net_second.h5")
        net_json = net.to_json()

        net_third = model_from_json(net_json)
        net_third.compile(optimizer=Adam(learning_rate = 10**-5), loss='mse', metrics=['acc'])
        # net_third.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_third.load_weights("net_second.h5")

    check = make_checkpoint(
        wdir+'third_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_third.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check])

    net_third.save(wdir+"model_"+str(num_rounds)+"r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
   


if __name__ == "__main__":

    # (0040,0000)->(8000,8000)->(8100,8102)->(8000,840a)->(850a,9520)
    first_stage( n=5*10**7,num_rounds=8,pairs=2)
    second_stage(n=5*10**7,num_rounds=8,pairs=2)
    stage_train( n=5*10**7,num_rounds=8,pairs=2)