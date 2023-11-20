from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv1D,  Input, Reshape, Permute, Add, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# import keras
from pickle import dump
import tensorflow as tf
import simon as si
import numpy as np
import multiprocessing as mp

bs = 5000
# wdir = './DATA_N_good_trained_nets/'
wdir = "./good_trained_nets/"
# 不断修改学习率


def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)


# #make residual tower of convolutional blocks
def make_resnet(pairs=2, num_blocks=2.5, num_filters=32, num_outputs=1,word_size=16, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):
    # Input and preprocessing layers
    # 生成初始数据输入形状（64）
    inp = Input(shape=(int(num_blocks * word_size * 2*pairs),))
    rs = Reshape((pairs, int(2*num_blocks),  word_size))(inp)
    perm = Permute((1, 3, 2))(rs)

    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv02 = Conv1D(num_filters, kernel_size=2, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv03 = Conv1D(num_filters, kernel_size=8, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    c2 = concatenate([conv01, conv02, conv03], axis=-1)
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0
    # 5*2=10层

    for i in range(depth):
        conv1 = Conv1D(num_filters*3, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters*3, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])  
        ks += 2
   
    dense0 = GlobalAveragePooling2D()(shortcut)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense0)
    model = Model(inputs=inp, outputs=out)
    return(model)


def train_simon_distinguisher(num_epochs, num_rounds=7, depth=1, pairs=1):
    print("pairs = ", pairs)
    # create the network
    print("num_rounds = ", num_rounds)

    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0"])
        # devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync

    with strategy.scope():
        net = make_resnet(pairs=pairs, depth=depth, reg_param=10**-5)
        # net.summary()
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # generate training and validation data
    # 生成训练数据
    # 使用的服务器核数
    process_number = 50
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(si.make_train_data, [(int(10**7/process_number),num_rounds,pairs,) for i in range(process_number)])

    X = accept_XY[0][0]
    Y = accept_XY[0][1]
    
    for i in range(process_number-1):
        X = np.concatenate((X,accept_XY[i+1][0]))
        Y = np.concatenate((Y,accept_XY[i+1][1]))
   
    with mp.Pool(process_number) as pool:
        accept_XY_eval = pool.starmap(si.make_train_data, [(int(10**6/process_number),num_rounds,pairs,) for i in range(process_number)])

    X_eval = accept_XY_eval[0][0]
    Y_eval = accept_XY_eval[0][1]
    
    for i in range(process_number-1):
        X_eval = np.concatenate((X_eval,accept_XY_eval[i+1][0]))
        Y_eval = np.concatenate((Y_eval,accept_XY_eval[i+1][1]))
  
    print("multiple processing end ......")

    # X, Y = si.make_train_data(int(10**7/pairs), num_rounds, pairs=pairs)
    # X_eval, Y_eval = si.make_train_data(int(10**6/pairs), num_rounds, pairs=pairs)
    # set up model checkpoint
    check = make_checkpoint(
        wdir+'best'+str(num_rounds)+'r_depth'+str(depth)+"_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+'.h5')
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    #train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size,
                validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    net.save(wdir+'simon_'+'model_'+str(num_rounds)+'r_depth'+str(depth) +
         "_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+'.h5')
         
    dump(h.history, open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth) +
         "_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+'.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    


if __name__ == "__main__":
    
    rounds = 11
    pairs = 1
    train_simon_distinguisher(num_epochs=20, num_rounds=rounds, depth=5, pairs=pairs)
    