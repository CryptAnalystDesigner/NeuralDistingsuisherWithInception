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
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():
        Z = net.predict(X, batch_size=batch_size)

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
    
    slices = 20

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

    pairs_list = [32]
    # pairs_list = [2,4,8,16,32]
    bs = 3000*32
    num = 3000
    # 读取模型网络参数
    wdir = "./our_train_net/"
    for pairs in pairs_list:


        # net8 = load_model(wdir+"model_8r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
        net8 = load_model(wdir+"transfer_learning_model_8r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
        m8,s8 = wrong_key_decryption(net=net8,bs=int(bs/pairs),n=num, pairs = pairs, diff=(0x0040, 0x0), nr=8)
        np.save(wdir+"transfer_learning_data_wrong_key_mean_8r_pairs"+str(pairs)+".npy",m8)
        np.save(wdir+"transfer_learning_data_wrong_key_std_8r_pairs"+str(pairs)+".npy",s8)
        
        # net7 = load_model(wdir+"model_7r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
        # m7,s7 = wrong_key_decryption(net=net7,bs=int(bs/pairs),n=num, pairs = pairs, diff=(0x0040, 0x0), nr=7)
        # np.save(wdir+"data_wrong_key_mean_7r_pairs"+str(pairs)+"_num"+str(num)+".npy",m7)
        # np.save(wdir+"data_wrong_key_std_7r_pairs"+str(pairs)+"_num"+str(num)+".npy",s7)

        # net6 = load_model(wdir+"model_6r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
        # m6,s6 = wrong_key_decryption(net=net6,bs=int(bs/pairs),n=num, pairs = pairs, diff=(0x0040, 0x0), nr=6)
        # np.save(wdir+"data_wrong_key_mean_6r_pairs"+str(pairs)+"_num"+str(num)+".npy",m6)
        # np.save(wdir+"data_wrong_key_std_6r_pairs"+str(pairs)+"_num"+str(num)+".npy",s6)