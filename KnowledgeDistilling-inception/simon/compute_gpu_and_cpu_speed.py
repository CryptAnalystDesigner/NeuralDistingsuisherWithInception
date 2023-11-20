import tensorflow as tf
import simon as si
import numpy as np
import os 

from time import time
from tensorflow.keras.models import load_model
import multiprocessing as mp
# tf.random.set_seed(1024)

        
# GPU 环境
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # 显示使用的显存
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8380)])

# 多核CPU 环境
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 单核CPU 环境
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1" 
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

pairs=8
num_rounds = 10


wdir = './teacher_distinguisher/'
net = load_model(wdir+"simon_model_10r_num_epochs20_acc_0.6886849999427795.h5")

# wdir = './student_distinguisher_knowledge_distillation/'
# net = load_model(wdir+"simon_model_10r_num_epochs40_acc_0.676828.h5")
# 生成训练数据
process_number = 50
with mp.Pool(process_number) as pool:
    accept_XY = pool.starmap(si.make_train_data, [(int(10**6/process_number),num_rounds,pairs,) for i in range(process_number)])
X = accept_XY[0][0]

for i in range(process_number-1):
    X = np.concatenate((X,accept_XY[i+1][0]))

print("multiple processing end ......")
# X, _ = si.make_train_data(10**6,num_rounds,pairs=pairs);
t2 = time()
Z = net.predict(X,batch_size=200)
t3 = time()
print("Run time of Teacher Model = ", t3 - t2)
