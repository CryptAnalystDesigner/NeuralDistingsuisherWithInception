import tensorflow as tf
import speck as sp
import numpy as np
import os 

from time import time
from tensorflow.keras.models import load_model

# tf.random.set_seed(1024)

        
# GPU 环境
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # 显示使用的显存
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)]
# )

# 多核CPU 环境
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 单核CPU 环境
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1" 
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

pairs=8
num_rounds = 7


# wdir = './teacher_distinguisher/'
# net = load_model(wdir+"speck_model_7r_num_epochs20_acc_0.8128569722175598.h5")
# X, _ = sp.make_train_data(10**7,num_rounds,pairs=pairs);

# t0 = time()
# Z = net.predict(X,batch_size=10000)
# t1 = time()
# print("Run time of Distilling Model =  ", t1 - t0)

wdir = './student_distinguisher/'
net = load_model(wdir+"speck_model_7r_num_epochs40_acc_0.805361.h5")

X, _ = sp.make_train_data(10**6,num_rounds,pairs=pairs);
t2 = time()
Z = net.predict(X,batch_size=400)
t3 = time()
print("Run time of Teacher Model = ", t3 - t2)
