from tensorflow.keras.regularizers import l2
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



bs = 1000
# wdir = './freshly_trained_nets/'
wdir = './good_trained_nets/'
# 不断修改学习率


def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)
# 回调函数 Callbacks 是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。然后，在模型上调用 fit() 函数时，
# 可以将 ModelCheckpoint 传递给训练过程


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)


# #make residual tower of convolutional blocks
def make_resnet(pairs=2, num_blocks=3, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):
    # Input and preprocessing layers
    # 生成初始数据输入形状（64）
    inp = Input(shape=(num_blocks * word_size * 2*pairs,))
    rs = Reshape((pairs, 2*num_blocks,  word_size))(inp)
    perm = Permute((1, 3, 2))(rs)

    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv02 = Conv1D(num_filters, kernel_size=3, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv03 = Conv1D(num_filters, kernel_size=5, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv04 = Conv1D(num_filters, kernel_size=7, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    c2 = concatenate([conv01, conv02, conv03, conv04], axis=-1)
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0
    # 5*2=10层

    for i in range(depth):
        conv1 = Conv1D(num_filters*4, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters*4, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2
    # add prediction head
    # 展开，全连接层
    flat1 = Flatten()(shortcut)
    dense0 = dropout(flat1, 0.8)
    dense0 = Dense(512, kernel_regularizer=l2(reg_param))(dense0)

    # dense0 = Dense(512, kernel_regularizer=l2(reg_param))(flat1)

    dense0 = BatchNormalization()(dense0)
    dense0 = Activation('relu')(dense0)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return(model)


def train_speck_distinguisher(num_epochs, num_rounds=7, depth=1, pairs=1):
    print("pairs = ", pairs)
    # create the network
    print("num_rounds = ", num_rounds)

    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync

    with strategy.scope():
        net = make_resnet(pairs=pairs, depth=depth, reg_param=10**-5)
        net.summary()
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # generate training and validation data
    # 生成训练数据
    X, Y = sp.make_train_data(int(10**7/8), num_rounds, pairs=pairs)
    X_eval, Y_eval = sp.make_train_data(int(10**6/8), num_rounds, pairs=pairs)
    # set up model checkpoint
    check = make_checkpoint(
        wdir+'best'+str(num_rounds)+'r_depth'+str(depth)+"_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+'.h5')
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    #train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size,
                validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    net.save(wdir+'model_'+str(num_rounds)+'r_depth'+str(depth) +
         "_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+'.h5')
         
    # np.save(wdir+'h'+str(num_rounds)+'r_depth' +
    #         str(depth)+"_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+"_val_acc"+'.npy', h.history['val_acc'])
    # np.save(wdir+'h'+str(num_rounds)+'r_depth' +
    #         str(depth)+"_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+"_val_loss"+'.npy', h.history['val_loss'])
    dump(h.history, open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth) +
         "_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+'.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    return(net, h)


if __name__ == "__main__":
    
    # rounds=[6,7]
    rounds=[7]
    pairs = 8
    for r in rounds:
        train_speck_distinguisher(num_epochs=40, num_rounds=r, depth=5, pairs=pairs)
