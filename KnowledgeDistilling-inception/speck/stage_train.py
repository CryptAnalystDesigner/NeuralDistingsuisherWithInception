import tensorflow as tf
from tensorflow.keras.models import load_model,model_from_json
from tensorflow.keras.optimizers import Adam
import os
import speck as sp
import numpy as np
import shutil


        
tf.random.set_seed(1024)

def stage_train(fn,num_rounds,num_epochs,pairs,batch_size):

    # wdir = "./teacher_distinguisher/"
    wdir = "./student_distinguisher/"
    if(not os.path.exists(wdir)):
        os.makedirs(wdir)
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    ]
    strategy = tf.distribute.MirroredStrategy(
            devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    
    batch_size = batch_size * strategy.num_replicas_in_sync

    orginal_file = wdir + fn
    # first stage
    with strategy.scope():
        
        net = load_model(orginal_file)
        net_json = net.to_json()
        net_first = model_from_json(net_json)
        net_first.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        net_first.load_weights(orginal_file)

    X, Y = sp.make_train_data(10**7,num_rounds-3,pairs=pairs,diff=(0x8000, 0x840a));
    X_eval, Y_eval = sp.make_train_data(10**6, num_rounds-3,pairs=pairs,diff=(0x8000, 0x840a));
    net_first.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=my_callbacks);
    net_first.save("./tmp/net_first.h5")
    
    # seconde stage
    with strategy.scope():
        
        net = load_model("./tmp/net_first.h5")
        net_json = net.to_json()
        net_second = model_from_json(net_json)
        net_second.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        net_second.load_weights("./tmp/net_first.h5")

    X, Y = sp.make_train_data(10**7,num_rounds,pairs=pairs);
    X_eval, Y_eval = sp.make_train_data(10**6, num_rounds,pairs=pairs);
    net_second.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=my_callbacks);
    net_second.save("./tmp/net_second.h5")

    # third stage
    with strategy.scope():
        
        net = load_model("./tmp/net_second.h5")
        net_json = net.to_json()
        net_third = model_from_json(net_json)
        net_third.compile(optimizer=Adam(learning_rate = 10**-5), loss='mse', metrics=['acc'])
        net_third.load_weights("./tmp/net_second.h5")

    X, Y = sp.make_train_data(10**7,num_rounds,pairs=pairs);
    X_eval, Y_eval = sp.make_train_data(10**6, num_rounds,pairs=pairs);
    h = net_third.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=my_callbacks);

    old_name = wdir+'speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)
    net_third.save(old_name+".h5")
    
    new_name = wdir+'speck_'+'model_'+str(num_rounds) + \
         "r_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))
     
    os.rename(old_name+".h5",new_name+".h5")
    
    if os.path.exists("./tmp/net_first.h5"):
        os.remove("./tmp/net_first.h5")
    if os.path.exists("./tmp/net_second.h5"):
        os.remove("./tmp/net_second.h5")
        

if __name__ == "__main__":
    
    # file_name = 'speck_model_7r_num_epochs20_acc_0.8128569722175598.h5'
    file_name = 'speck_model_7r_num_epochs40_acc_0.805361.h5'
    # file_name = 'speck_model_7r_num_epochs40_acc_0.799585.h5'
    stage_train(fn=file_name,num_rounds=8,num_epochs=10,pairs=8,batch_size=3000)