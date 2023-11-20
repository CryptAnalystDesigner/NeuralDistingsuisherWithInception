from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Dense, Conv1D,GlobalAveragePooling2D,GlobalAveragePooling1D,MaxPooling2D,Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model,load_model
from tensorflow.nn import dropout
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from pickle import dump
import tensorflow as tf
import simon as si
import numpy as np
import shutil
import os
import multiprocessing as mp


from tensorflow import keras
from tensorflow.keras import layers


def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)
        
def TeacherModel(pairs=2, num_blocks=2.5, num_filters=32, num_outputs=1, word_size=16, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):

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
    return(model);

def StudentModel(pairs=2, num_blocks=2.5, num_filters=32, num_outputs=1, word_size=16, ks=3, depth=5, reg_param=0.0001,final_activation='sigmoid'):

    inp = Input(shape=(int(num_blocks * word_size * 2*pairs),))
    rs = Reshape((pairs, int(2*num_blocks),  word_size))(inp)
    perm = Permute((1, 3, 2))(rs)
    # conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
    #                 kernel_regularizer=l2(reg_param))(perm)
    # conv02 = Conv1D(num_filters, kernel_size=3, padding='same',
    #                 kernel_regularizer=l2(reg_param))(perm)
    # conv03 = Conv1D(num_filters, kernel_size=5, padding='same',
    #                 kernel_regularizer=l2(reg_param))(perm)
    conv04 = Conv1D(num_filters, kernel_size=8, padding='same',
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
    
    dense0 = GlobalAveragePooling2D()(shortcut)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense0)
    model = Model(inputs=inp, outputs=out)
    return(model);


def teacher_distinguisher_train(num_rounds,num_epochs,pairs,depth,batch_size):

    wdir = "./teacher_distinguisher/"
    if(not os.path.exists(wdir)):
            os.makedirs(wdir)
            
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
    
    old_name = wdir+'simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)
         
    check = make_checkpoint(old_name+'.h5')
    
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = batch_size * strategy.num_replicas_in_sync
   
    with strategy.scope():
        model = TeacherModel(pairs=pairs,depth=depth,reg_param=10**-5)   
        # model.summary()
        model.compile(optimizer='adam',loss='mse',metrics=['acc']);
    
    h = model.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=[lr, check]);
    dump(h.history,open(wdir+'history_simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))+'.p','wb'));   
    
    new_name = wdir+'simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))
         
    os.rename(old_name+'.h5',new_name+'.h5') 
    print("Best validation accuracy: ", np.max(h.history['val_acc']));

def student_distinguisher_train(num_rounds,num_epochs,pairs,depth,batch_size):

    wdir = "./student_distinguisher/"
    if(not os.path.exists(wdir)):
            os.makedirs(wdir)
            
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
    
    old_name = wdir+'simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)  
    check = make_checkpoint(old_name+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = batch_size * strategy.num_replicas_in_sync
   
    with strategy.scope():
        model = StudentModel(pairs=pairs,depth=depth,reg_param=10**-5)   
        # model.summary()
        model.compile(optimizer='adam',loss='mse',metrics=['acc']);
        
    
    h = model.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=[lr, check]);
    dump(h.history,open(wdir+'history_simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))+'.p','wb'));
    
    
    new_name = wdir+'simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))
         
    os.rename(old_name+'.h5',new_name+'.h5') 
    print("Best validation accuracy: ", np.max(h.history['val_acc']));

def student_distinguisher_knowledge_distillation(Teacher_fn,num_rounds,num_epochs,pairs,depth,batch_size):

    wdir = "./student_distinguisher_knowledge_distillation/"


    if(not os.path.exists(wdir)):
            os.makedirs(wdir)
    # 使用Teacher模型制作标签
    net = load_model(Teacher_fn)
   # 生成训练数据
    # 使用的服务器核数
    process_number = 50
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(si.make_train_data, [(int(10**7/process_number),num_rounds,pairs,) for i in range(process_number)])

    X = accept_XY[0][0]
 
    
    for i in range(process_number-1):
        X = np.concatenate((X,accept_XY[i+1][0]))
   
    with mp.Pool(process_number) as pool:
        accept_XY_eval = pool.starmap(si.make_train_data, [(int(10**6/process_number),num_rounds,pairs,) for i in range(process_number)])

    X_eval = accept_XY_eval[0][0]
    
    for i in range(process_number-1):
        X_eval = np.concatenate((X_eval,accept_XY_eval[i+1][0]))
  
    print("multiple processing end ......")
    
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
    
    old_name = wdir+'simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)  
    check = make_checkpoint(old_name+'.h5')
    lr = LearningRateScheduler(cyclic_lr(20,0.001, 0.0001));
    with strategy.scope():
        model = StudentModel(pairs=pairs,depth=depth,reg_param=10**-5)   
        model.summary()
        model.compile(optimizer='adam',loss='mse',metrics=['acc']);

    
    h = model.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=[lr, check]);
    
    X, Y = si.make_train_data(10**6,num_rounds,pairs=pairs);
    with strategy.scope():
        Z = model.predict(X,batch_size=batch_size).flatten();
    Zbin = (Z >= 0.5);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr);
    
    dump(h.history,open(wdir+'history_simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(acc)+'.p','wb'));
    
    new_name = wdir+'simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(acc)
         
    os.rename(old_name+'.h5',new_name+'.h5') 
    print("Best validation accuracy: ", np.max(acc));
    

    
if __name__ == "__main__":
    
    rounds=[10]
    pairs = 8
    batch_size = 5000
    
    # 直接训练Teacher模型
    # for r in rounds:
    #     teacher_distinguisher_train(num_rounds=r, num_epochs=20, pairs=pairs, depth=5,batch_size=batch_size)
        
    # 直接训练student模型
    # for r in rounds:
    #     student_distinguisher_train(num_rounds=r, num_epochs=20, pairs=pairs, depth=5,batch_size=batch_size)

        
    # 训练蒸馏Student模型
    file_name = './teacher_distinguisher/'+'simon_model_10r_num_epochs20_acc_0.6886849999427795.h5'
    for r in rounds:
        student_distinguisher_knowledge_distillation(Teacher_fn=file_name, num_rounds=r, num_epochs=40, pairs=pairs, depth=5,batch_size=batch_size)
