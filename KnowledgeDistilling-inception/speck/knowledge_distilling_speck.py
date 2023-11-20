from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Dense, Conv1D,GlobalAveragePooling2D,GlobalAveragePooling1D,MaxPooling2D,Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model,load_model
from tensorflow.nn import dropout
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from pickle import dump
import tensorflow as tf
import speck as sp
import numpy as np
import shutil
import os


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

tf.random.set_seed(1024)
def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

        
def TeacherModel(pairs=2, num_blocks=3, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=5, reg_param=0.0001):

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
   
    dense0 = Flatten()(shortcut)
 
    dense0 = Dense(512, kernel_regularizer=l2(reg_param))(dense0)
    dense0 = BatchNormalization()(dense0)
    dense0 = Activation('relu')(dense0)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    
    x = Dense(num_outputs)(dense2);
    out = tf.nn.sigmoid(x)
    model = Model(inputs=inp, outputs=out);
    return(model);

def StudentModel(pairs=2, num_blocks=3, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=5, reg_param=0.0001,final_activation='sigmoid'):

    inp = Input(shape=(num_blocks * word_size * 2*pairs,))
    rs = Reshape((pairs, 2*num_blocks,  word_size))(inp)
    perm = Permute((1, 3, 2))(rs)
    # conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
    #                 kernel_regularizer=l2(reg_param))(perm)
    # conv02 = Conv1D(num_filters, kernel_size=3, padding='same',
    #                 kernel_regularizer=l2(reg_param))(perm)
    # conv03 = Conv1D(num_filters, kernel_size=5, padding='same',
    #                 kernel_regularizer=l2(reg_param))(perm)
    conv04 = Conv1D(num_filters, kernel_size=7, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
 
    # c2 = concatenate([conv01,conv04], axis=-1)
    # conv0 = BatchNormalization()(c2)
    
    conv0 = BatchNormalization()(conv04)
    
    conv0 = Activation('relu')(conv0)
    shortcut = conv0

    for i in range(depth):
        conv1 = Conv1D(num_filters*1, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters*1, kernel_size=ks,padding='same', 
                       kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        # ks += 2
   
    # dense0 = Flatten()(shortcut)
    # dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Activation('relu')(dense1)
    # dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    # dense2 = BatchNormalization()(dense2)
    # dense2 = Activation('relu')(dense2)
    # x = Dense(num_outputs)(dense2);
    
    dense0 = GlobalAveragePooling2D()(shortcut) 
    x = Dense(num_outputs)(dense0)   
    out = tf.nn.sigmoid(x)
    model = Model(inputs=inp, outputs=out);
    return(model);


def teacher_distinguisher_train(num_rounds,num_epochs,pairs,depth,batch_size):

    wdir = "./teacher_distinguisher/"
    if(not os.path.exists(wdir)):
            os.makedirs(wdir)
            
    X, Y = sp.make_train_data(10**7,num_rounds,pairs=pairs);
    X_eval, Y_eval = sp.make_train_data(10**6, num_rounds,pairs=pairs);
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = batch_size * strategy.num_replicas_in_sync
   
    with strategy.scope():
        model = TeacherModel(pairs=pairs,depth=5,reg_param=10**-5)   
        model.summary()
        model.compile(optimizer='adam',loss='mse',metrics=['acc']);
        
    my_callbacks = [
        lr,
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    ]
    
    h = model.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=my_callbacks);
    dump(h.history,open(wdir+'history_speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))+'.p','wb'));
    
    old_name = wdir+'speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)
    model.save(old_name+'.h5')
    
    new_name = wdir+'speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))
         
    os.rename(old_name+'.h5',new_name+'.h5') 
    print("Best validation accuracy: ", np.max(h.history['val_acc']));

def student_distinguisher_train(num_rounds,num_epochs,pairs,depth,batch_size):

    wdir = "./student_distinguisher/"
    if(not os.path.exists(wdir)):
            os.makedirs(wdir)
            
    X, Y = sp.make_train_data(10**7,num_rounds,pairs=pairs);
    X_eval, Y_eval = sp.make_train_data(10**6, num_rounds,pairs=pairs);
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = batch_size * strategy.num_replicas_in_sync
   
    with strategy.scope():
        model = StudentModel(pairs=pairs,depth=depth,reg_param=10**-5)   
        model.summary()
        model.compile(optimizer='adam',loss='mse',metrics=['acc']);
        
    my_callbacks = [
        lr,
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    ]
    
    h = model.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=my_callbacks);
    dump(h.history,open(wdir+'history_speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))+'.p','wb'));
    
    old_name = wdir+'speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)
    model.save(old_name+'.h5')
    
    new_name = wdir+'speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))
         
    os.rename(old_name+'.h5',new_name+'.h5') 
    print("Best validation accuracy: ", np.max(h.history['val_acc']));

def student_distinguisher_knowledge_distillation(Teacher_fn,num_rounds,num_epochs,pairs,depth,batch_size):

    wdir = "./student_distinguisher/"
    # wdir = "./student_distinguisher_mixed_precision/"
    if(not os.path.exists(wdir)):
            os.makedirs(wdir)
    # 使用Teacher模型制作标签
    net = load_model(Teacher_fn)
    X, _ = sp.make_train_data(10**7,num_rounds,pairs=pairs);
    X_eval, _ = sp.make_train_data(10**6,num_rounds,pairs=pairs);
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = batch_size * strategy.num_replicas_in_sync
    with strategy.scope():
        Z = net.predict(X,batch_size=batch_size).flatten();
    Y = (Z >= 0.5); 
    with strategy.scope():
        Z_eval = net.predict(X_eval,batch_size=batch_size).flatten();
    Y_eval = (Z_eval >= 0.5); 
    
    print("Lable generation finished")
    
    # # 设置半精度，加速模型训练
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    
    lr = LearningRateScheduler(cyclic_lr(20,0.001, 0.0001));
    with strategy.scope():
        model = StudentModel(pairs=pairs,depth=depth,reg_param=10**-5)   
        model.summary()
        model.compile(optimizer='adam',loss='mse',metrics=['acc']);
        
    my_callbacks = [
        lr,
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    ]
    
    h = model.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=my_callbacks);
    
    X, Y = sp.make_train_data(10**6,num_rounds,pairs=pairs);
    with strategy.scope():
        Z = model.predict(X,batch_size=batch_size).flatten();
    Zbin = (Z >= 0.5);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr);
    
    dump(h.history,open(wdir+'history_speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(acc)+'.p','wb'));
    
    old_name = wdir+'speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)
    model.save(old_name+'.h5')
    
    new_name = wdir+'speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(acc)
         
    os.rename(old_name+'.h5',new_name+'.h5') 
    print("Best validation accuracy: ", np.max(acc));
    

    
if __name__ == "__main__":
    
    rounds=[7]
    pairs = 8
    batch_size = 3000
    
    # 直接训练Teacher模型
    # for r in rounds:
    #     teacher_distinguisher_train(num_rounds=r, num_epochs=20, pairs=pairs, depth=5,batch_size=batch_size)
        
    # 直接训练student模型
    for r in rounds:
        student_distinguisher_train(num_rounds=r, num_epochs=20, pairs=pairs, depth=5,batch_size=batch_size)
        
        
    # 训练蒸馏Student模型
    # file_name = './teacher_distinguisher/'+'speck_model_7r_num_epochs20_acc_0.8128569722175598.h5'
    # for r in rounds:
    #     student_distinguisher__knowledge_distillation(Teacher_fn=file_name, num_rounds=r, num_epochs=40, pairs=pairs, depth=5,batch_size=batch_size)
