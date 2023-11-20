from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model,model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# import keras
from pickle import dump
import tensorflow as tf
import simon as si
import numpy as np
import multiprocessing as mp
import os 
    

def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)



def stage_train(fn,num_rounds,num_epochs,pairs,batch_size):

    # wdir = "./teacher_distinguisher/"
    wdir = "./student_distinguisher_knowledge_distillation/"
    if(not os.path.exists(wdir)):
        os.makedirs(wdir)

    strategy = tf.distribute.MirroredStrategy(
            devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    
    batch_size = batch_size * strategy.num_replicas_in_sync

    # orginal_file = wdir + fn
    # # first stage
    # with strategy.scope():
        
    #     net = load_model(orginal_file)
    #     net_json = net.to_json()
    #     net_first = model_from_json(net_json)
    #     net_first.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
    #     net_first.load_weights(orginal_file)

    # # 生成训练数据
    # # 使用的服务器核数
    # process_number = 50
    # with mp.Pool(process_number) as pool:
    #     accept_XY = pool.starmap(si.make_train_data, [(int(10**7/process_number),num_rounds-3,pairs,(0x0440, 0x0100),) for i in range(process_number)])

    # X = accept_XY[0][0]
    # Y = accept_XY[0][1]

    # for i in range(process_number-1):
    #     X = np.concatenate((X,accept_XY[i+1][0]))
    #     Y = np.concatenate((Y,accept_XY[i+1][1]))

    # with mp.Pool(process_number) as pool:
    #     accept_XY_eval = pool.starmap(si.make_train_data, [(int(10**6/process_number),num_rounds-3,pairs,(0x0440, 0x0100),) for i in range(process_number)])

    # X_eval = accept_XY_eval[0][0]
    # Y_eval = accept_XY_eval[0][1]
    
    # for i in range(process_number-1):
    #     X_eval = np.concatenate((X_eval,accept_XY_eval[i+1][0]))
    #     Y_eval = np.concatenate((Y_eval,accept_XY_eval[i+1][1]))

    # print("multiple processing end ......")
    
    # check = make_checkpoint("./tmp/net_first.h5")
    # net_first.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=[check]);

    
    # seconde stage
    # with strategy.scope():
        
    #     net = load_model("./tmp/net_first.h5")
    #     net_json = net.to_json()
    #     net_second = model_from_json(net_json)
    #     net_second.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
    #     net_second.load_weights("./tmp/net_first.h5")

    # # 生成训练数据
    # # 使用的服务器核数
    # process_number = 50
    # with mp.Pool(process_number) as pool:
    #     accept_XY = pool.starmap(si.make_train_data, [(int(10**7/process_number),num_rounds,pairs,) for i in range(process_number)])

    # X = accept_XY[0][0]
    # Y = accept_XY[0][1]

    # for i in range(process_number-1):
    #     X = np.concatenate((X,accept_XY[i+1][0]))
    #     Y = np.concatenate((Y,accept_XY[i+1][1]))

    # with mp.Pool(process_number) as pool:
    #     accept_XY_eval = pool.starmap(si.make_train_data, [(int(10**6/process_number),num_rounds,pairs,) for i in range(process_number)])

    # X_eval = accept_XY_eval[0][0]
    # Y_eval = accept_XY_eval[0][1]
    
    # for i in range(process_number-1):
    #     X_eval = np.concatenate((X_eval,accept_XY_eval[i+1][0]))
    #     Y_eval = np.concatenate((Y_eval,accept_XY_eval[i+1][1]))

    # print("multiple processing end ......")
    # check = make_checkpoint("./tmp/net_second.h5")
    # net_second.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=[check]);


    # third stage
    with strategy.scope():
        
        net = load_model("./tmp/net_second.h5")
        net_json = net.to_json()
        net_third = model_from_json(net_json)
        net_third.compile(optimizer=Adam(learning_rate = 10**-5), loss='mse', metrics=['acc'])
        net_third.load_weights("./tmp/net_second.h5")

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
    check = make_checkpoint(old_name+".h5")
    h = net_third.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=[check]);

    new_name = wdir+'simon_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))
     
    os.rename(old_name+".h5",new_name+".h5")
    
    if os.path.exists("./tmp/net_first.h5"):
        os.remove("./tmp/net_first.h5")
    if os.path.exists("./tmp/net_second.h5"):
        os.remove("./tmp/net_second.h5")
        

if __name__ == "__main__":
    

    file_name = 'simon_model_11r_num_epochs40_acc_0.558271.h5'

    stage_train(fn=file_name,num_rounds=12,num_epochs=10,pairs=8,batch_size=3000)