import sys
import simon as si
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


WORD_SIZE = si.WORD_SIZE()


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
    
   
    ctdata0l,ctdata0r = si.dec_one_round((ctdata0l, ctdata0r), keys)
    ctdata1l,ctdata1r = si.dec_one_round((ctdata1l, ctdata1r), keys)
    
    
    R0 = si.rol(ctdata0r,8)&si.rol(ctdata0r,1)^si.rol(ctdata0r,2)^ctdata0l
    R1 = si.rol(ctdata1r,8)&si.rol(ctdata1r,1)^si.rol(ctdata1r,2)^ctdata1l
    X = si.convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r,R0^R1])

    X = X.reshape(num_key,pairs,16*5)
    X = X.reshape(num_key,pairs*16*5)
     
    return X

def predict(X, net,bs):

 
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():
        Z = net.predict(X, batch_size=batch_size)

    return Z

def wrong_key_decryption(net,bs,n=3000, pairs = 2, diff=(0x0000,0x0040), nr=7):
    
    # 生成需要测试的明文和密文
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16)
    keys = np.repeat(keys,pairs).reshape(4,-1)
    pt0l = np.frombuffer(urandom(2*n*pairs), dtype=np.uint16)
    pt0r = np.frombuffer(urandom(2*n*pairs), dtype=np.uint16)
    pt1l = pt0l ^ diff[0]
    pt1r = pt0r ^ diff[1]
    ks = si.expand_key(keys, nr+1)
    # 生成nr+1轮密文，然后用不同密钥进行解密
    ct0l, ct0r = si.encrypt((pt0l, pt0r), ks)
    ct1l, ct1r = si.encrypt((pt1l, pt1r), ks)
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
        X = X.reshape(int(n/slices)*num_keys,pairs*80)
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


    rounds = 9
    bs = 3000*32
    num = 3000
    pairs_list = [8]
    # 读取模型网络参数
    wdir = "../good_trained_nets/"
    for pairs in pairs_list:

        net = load_model(wdir+"simon_model_"+str(rounds)+"r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
        m,s = wrong_key_decryption(net=net,bs=int(bs/pairs),n=num, pairs = pairs, diff=(0x0000,0x0040), nr=rounds)
        np.save("simon_data_wrong_key_mean_"+str(rounds)+"r_pairs"+str(pairs)+".npy",m)
        np.save("simon_data_wrong_key_std_"+str(rounds)+"r_pairs"+str(pairs)+".npy",s)
     