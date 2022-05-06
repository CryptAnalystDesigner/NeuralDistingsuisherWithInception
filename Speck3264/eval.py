import speck as sp
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable = True) 

# wdir = './good_trained_nets/'
wdir = './freshly_trained_nets/'
pairs = 8
# net5= load_model(wdir+'model_'+str(5)+'r_depth'+str(5) +"_num_epochs"+str(20)+"_pairs"+str(pairs)+'.h5')
# net6= load_model(wdir+'model_'+str(6)+'r_depth'+str(5) +"_num_epochs"+str(20)+"_pairs"+str(pairs)+'.h5')
# net7= load_model(wdir+'model_'+str(7)+'r_depth'+str(5) +"_num_epochs"+str(20)+"_pairs"+str(pairs)+'.h5')
# net8= load_model(wdir+'model_'+str(8)+'r_depth'+str(5) +"_num_epochs"+str(20)+"_pairs"+str(pairs)+'.h5')

net8= load_model(wdir+"n=1000000model_8r_depth5_num_epochs20_pairs8_num_keys10000.h5")


def evaluate(net,X,Y):
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = 2000 * strategy.num_replicas_in_sync
    with strategy.scope():
        Z = net.predict(X,batch_size=batch_size).flatten();
    Zbin = (Z >= 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);

# X5,Y5 = sp.make_train_data(5*10**6,5,pairs);
# X6,Y6 = sp.make_train_data(10**6,6,pairs);
# X7,Y7 = sp.make_train_data(10**6,7,pairs);
X8,Y8 = sp.make_train_data(10**6,8,pairs);

# X5r, Y5r = sp.real_differences_data(5*10**6,5,pairs);
# X6r, Y6r = sp.real_differences_data(10**6,6,pairs);
# X7r, Y7r = sp.real_differences_data(10**6,7,pairs);
X8r, Y8r = sp.real_differences_data(10**6,8,pairs);

print('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting');
# print('5 rounds:');
# evaluate(net5, X5, Y5);
# print('6 rounds:');
# evaluate(net6, X6, Y6);
# print('7 rounds:');
# evaluate(net7, X7, Y7);
print('8 rounds:');
evaluate(net8, X8, Y8);

print('\nTesting real differences setting now.');
# print('5 rounds:');
# evaluate(net5, X5r, Y5r);
# print('6 rounds:');
# evaluate(net6, X6r, Y6r);
# print('7 rounds:');
# evaluate(net7, X7r, Y7r);
print('8 rounds:');
evaluate(net8, X8r, Y8r);
