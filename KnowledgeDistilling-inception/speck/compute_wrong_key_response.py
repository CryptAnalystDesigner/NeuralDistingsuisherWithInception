import sys
import speck as sp
import numpy as np

from tensorflow.keras.models import load_model
import tensorflow as tf 
from scipy.stats import norm
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2
import multiprocessing as mp
import gc

WORD_SIZE = sp.WORD_SIZE()


def key_average(pairs, ctdata0l, ctdata0r, ctdata1l, ctdata1r, ks_nr):

    # print("############")
    rsubkeys = np.arange(0, 2**WORD_SIZE, dtype=np.uint16)
    keys = rsubkeys ^ ks_nr

    num_key = len(keys)

    ctdata0l = np.tile(ctdata0l,num_key)
    ctdata0r = np.tile(ctdata0r,num_key)
    ctdata1l = np.tile(ctdata1l,num_key)
    ctdata1r = np.tile(ctdata1r,num_key)

    keys = np.repeat(keys,pairs)
    
   
    ctdata0l,ctdata0r = sp.dec_one_round((ctdata0l, ctdata0r), keys)
    ctdata1l,ctdata1r = sp.dec_one_round((ctdata1l, ctdata1r), keys)
    
    
    R0 = sp.ror(ctdata0l ^ ctdata0r, sp.BETA())
    R1 = sp.ror(ctdata1l ^ ctdata1r, sp.BETA())
    X = sp.convert_to_binary([R0, R1, ctdata0l, ctdata0r, ctdata1l, ctdata1r])

    X = X.reshape(num_key,pairs,16*6)
    X = X.reshape(num_key,pairs*16*6)
     
    return X

def predict(X, net,bs):

 
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    # print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():
        Z = net.predict(X, batch_size=batch_size)
    # print("Z.shape = ",Z.shape)
    return Z

def wrong_key_decryption(net,bs,n=3000, pairs = 2, diff=(0x0040, 0x0), nr=7):
    
    
    # 生成需要测试的明文和密文
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16)
    # keys = np.squeeze(np.tile(keys.reshape(-1, 1), (1, pairs)
    #                           ).reshape(-1, 1)).reshape(4, -1)
    keys = np.repeat(keys,pairs).reshape(4,-1)
    pt0l = np.frombuffer(urandom(2*n*pairs), dtype=np.uint16)
    pt0r = np.frombuffer(urandom(2*n*pairs), dtype=np.uint16)
    pt1l = pt0l ^ diff[0]
    pt1r = pt0r ^ diff[1]
    ks = sp.expand_key(keys, nr+1)
    # 生成nr+1轮密文，然后用不同密钥进行解密
    ct0l, ct0r = sp.encrypt((pt0l, pt0r), ks)
    ct1l, ct1r = sp.encrypt((pt1l, pt1r), ks)
    print("ks[nr] shape",ks[nr].shape)
    
    slices = 10

    ct0l = ct0l.reshape(slices, -1)
    ct0r = ct0r.reshape(slices, -1)
    ct1l = ct1l.reshape(slices, -1)
    ct1r = ct1r.reshape(slices, -1)

    nr_key = np.copy(ks[nr])
    nr_key = nr_key.reshape(slices,-1)

    # 开始执行多进程
    process_number = mp.cpu_count()-4
  
    Z = []
    for i in range(slices):

        pool = mp.Pool(process_number)
        X = pool.starmap(key_average, [
                             (pairs, ct0l[i][j*pairs:(j+1)*pairs], ct0r[i][j*pairs:(j+1)*pairs], \
             ct1l[i][j*pairs:(j+1)*pairs], ct1r[i][j*pairs:(j+1)*pairs], nr_key[i][j*pairs],) for j in range(int(n/slices))])
        print("multiple processing end ......")

        pool.close()
        pool.join()

        X = np.array(X).flatten()
        
        num_keys=2**16
        # X = X.reshape(n*num_keys,pairs*96)
        X = X.reshape(int(n/slices)*num_keys,pairs*96)
        Z.append(predict(X, net,bs))
        # print("len(Z) = ",len(Z))
        del X 
        gc.collect()

    Z = np.array(Z).flatten()
    Z = Z.reshape(n,-1)

    mean = np.mean(Z,axis=0)
    std = np.std(Z,axis=0)

    print("mean shape",mean.shape)
    print("std shape",std.shape)


    return mean, std


if __name__ == "__main__":
   
    # pairs=8;bs = 10000
    # num = 3000;r = 7
    # # 读取模型网络参数
    # wdir = './teacher_distinguisher/'
    # target = './wkrp/'
    # file_name = "speck_model_7r_num_epochs20_acc_0.8128569722175598"\
    # # file_name = "speck_model_7r_num_epochs40_acc_0.805361"
    # file_name = "speck_model_7r_num_epochs40_acc_0.799585"
    # net = load_model(wdir+file_name+".h5")
    # m,s = wrong_key_decryption(net=net,bs=bs,n=num, pairs = pairs, nr=r)
    # np.save(target+file_name+"_data_wrong_key_mean_"+str(r)+"r_pairs"+str(pairs)+".npy",m)
    # np.save(target+file_name+"_data_wrong_key_std_"+str(r)+"r_pairs"+str(pairs)+".npy",s)
    
    pairs=8;bs = 10000
    num = 3000;r = 8

    # # 读取模型网络参数
    wdir = './student_distinguisher/'
    target = './wkrp/'
    # file_name = "speck_model_8r_num_epochs10_acc_0.5532829761505127"
    # file_name = "speck_model_8r_num_epochs10_acc_0.5416589975357056"
    file_name = "speck_model_8r_num_epochs10_acc_0.540369987487793"
    net = load_model(wdir+file_name+".h5")
    m,s = wrong_key_decryption(net=net,bs=bs,n=num, pairs = pairs, nr=r)
    np.save(target+file_name+"_data_wrong_key_mean_"+str(r)+"r_pairs"+str(pairs)+".npy",m)
    np.save(target+file_name+"_data_wrong_key_std_"+str(r)+"r_pairs"+str(pairs)+".npy",s)