import simon as si
import numpy as np
import multiprocessing as mp
import tensorflow as tf 
from tensorflow.keras.models import load_model

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable = True) 


pairs = 8
nr = 10
# wdir = './teacher_distinguisher/'
# net= load_model(wdir+'simon_model_11r_num_epochs20_acc_0.5592020153999329.h5')
# net= load_model(wdir+'simon_model_10r_num_epochs20_acc_0.6886849999427795.h5')

wdir = './student_distinguisher/'
net= load_model(wdir+'simon_model_10r_num_epochs20_acc_0.6803439855575562.h5')

# wdir = './student_distinguisher_knowledge_distillation/'
# net= load_model(wdir+'simon_model_10r_num_epochs40_acc_0.676828.h5')

def evaluate(net,X,Y):

    batch_size = 2000 
    Z = net.predict(X,batch_size=batch_size).flatten();
    Zbin = (Z >= 0.5);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;

    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr);


process_number = 50
with mp.Pool(process_number) as pool:
    accept_XY = pool.starmap(si.make_train_data, [(int(10**6/process_number),nr,pairs,) for i in range(process_number)])
X = accept_XY[0][0]
Y = accept_XY[0][1]

for i in range(process_number-1):
    X = np.concatenate((X,accept_XY[i+1][0]))
    Y = np.concatenate((Y,accept_XY[i+1][1]))


evaluate(net, X, Y);


