# Proof of concept implementation of 11-round key recovery attack

import simon as si
import numpy as np

from tensorflow.keras.models import load_model
from os import urandom
from math import sqrt, log2
from time import time
from math import log2
import multiprocessing as mp
import tensorflow as tf 


WORD_SIZE = si.WORD_SIZE()



num_neural_bit = 11

# 读取模型网络参数
mswdir = './wkrp/'
rounds = 12
pairs = 8


net12 = None
net11 = None

m12 = np.load(mswdir+"simon_model_12r_num_epochs10_acc_0.5086619853973389_data_wrong_key_mean_12r_pairs8.npy");
s12 = np.load(mswdir+"simon_model_12r_num_epochs10_acc_0.5086619853973389_data_wrong_key_std_12r_pairs8.npy"); 
m11 = np.load(mswdir+"simon_model_11r_num_epochs40_acc_0.558271_data_wrong_key_mean_11r_pairs8.npy");
s11 = np.load(mswdir+"simon_model_11r_num_epochs40_acc_0.558271_data_wrong_key_std_11r_pairs8.npy"); 


s12 = 1.0/s12
s11 = 1.0/s11


# binarize a given ciphertext sample
# ciphertext is given as a sequence of arrays
# each array entry contains one word of ciphertext for all ciphertexts given


def convert_to_binary(l):
    n = len(l)
    k = WORD_SIZE * n
    X = np.zeros((k, len(l[0])), dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE
        offset = WORD_SIZE - 1 - i % WORD_SIZE
        X[i] = (l[index] >> offset) & 1
    X = X.transpose()
    return(X)

# 汉明重量
def hw(v):
    res = np.zeros(v.shape, dtype=np.uint8)
    for i in range(16):
        res = res + ((v >> i) & 1)
    return(res)


# 将初始输入差分的wt值设置低一些。2^16-1
low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16)
# 真实密钥和猜测密钥的wt值相差不能超过2
low_weight = low_weight[hw(low_weight) <= 2]

# make a plaintext structure
# takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits


def make_structure(pt0, pt1, diff=(0x0440,0x1000), neutral_bits=[[20],[21],[22],[9,16],[2,11,25],[14],[15],[6,29]]):
    # p0和p1是分别是随机生成明文的左右两边
    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    for subset in neutral_bits:
        d0_sum = 0x0
        d1_sum = 0x0
        for i in subset:
            d = 1 << i
            # d0影响高位，d1控制低位
            d0 = d >> 16
            d1 = d & 0xffff

            d0_sum = d0_sum ^ d0
            d1_sum = d1_sum ^ d1
        p0 = np.concatenate([p0, p0 ^ d0_sum], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1_sum], axis=1)
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    return(p0, p1, p0b, p1b)



def gen_key(nr):
    # 就是只要一个密钥
    key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4,-1)
    ks = si.expand_key(key, nr)
    
    return(ks)

def test_correct_pairs(pt0l,pt0r,key, nr=4,diff=(0x0440,0x1000),target_diff=(0x0000, 0x0040)):
    
    
    pt1l = pt0l ^ diff[0]
    pt1r = pt0r ^ diff[1]
    pt0l_1, pt0r_1 = si.dec_one_round((pt0l,pt0r), 0)
    pt1l_1, pt1r_1 = si.dec_one_round((pt1l, pt1r), 0)

    ct0l, ct0r = si.encrypt((pt0l_1, pt0r_1 ), key[:nr])
    ct1l, ct1r = si.encrypt((pt1l_1, pt1r_1), key[:nr])
    diff0 = ct0l ^ ct1l
    diff1 = ct0r ^ ct1r
    
    d0 = (diff0 == target_diff[0])
    d1 = (diff1 == target_diff[1])
    d = d0 * d1
   
    return(d)

def gen_plain(n):
    
    pt0 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    pt1 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    return(pt0, pt1)

def gen_challenge(pt0,pt1, key, diff=(0x0440,0x1000), neutral_bits=[20, 21, 22, 14, 15, 23]):
    #           明文对的数量 轮数          差分             中立bit                           密钥扩展算法

    pt0a, pt1a, pt0b, pt1b = make_structure(
        pt0, pt1, diff=diff, neutral_bits=neutral_bits)
    pt0a, pt1a = si.dec_one_round((pt0a, pt1a), 0)
    pt0b, pt1b = si.dec_one_round((pt0b, pt1b), 0)
    # 加密了nr轮
    ct0a, ct1a = si.encrypt((pt0a, pt1a), key)
    ct0b, ct1b = si.encrypt((pt0b, pt1b), key)
    return([ct0a, ct1a, ct0b, ct1b])


# having a good key candidate, exhaustively explore all keys with hamming distance less than two of this key

def verifier_search(pairs,cts, best_guess, use_n=2**num_neural_bit, net=net12):
    # 进来的数据就是一个密文结构内的数据
    # print("verifier search......")
    # print(best_guess);
    # 控制最多wt值相差2
    # ck1数量为137
    ck1 = best_guess[0] ^ low_weight
    ck2 = best_guess[1] ^ low_weight

    n = len(ck1)
    ck1 = np.repeat(ck1, n)
    keys1 = np.copy(ck1)

    ck2 = np.tile(ck2, n)
    keys2 = np.copy(ck2)

    ck1 = np.repeat(ck1, pairs*use_n)
    ck2 = np.repeat(ck2, pairs*use_n)

    ct0a = np.tile(cts[0].transpose().flatten(), n*n)
    ct1a = np.tile(cts[1].transpose().flatten(), n*n)
    ct0b = np.tile(cts[2].transpose().flatten(), n*n)
    ct1b = np.tile(cts[3].transpose().flatten(), n*n)
    pt0a, pt1a = si.dec_one_round((ct0a, ct1a), ck1)
    pt0b, pt1b = si.dec_one_round((ct0b, ct1b), ck1)
    pt0a, pt1a = si.dec_one_round((pt0a, pt1a), ck2)
    pt0b, pt1b = si.dec_one_round((pt0b, pt1b), ck2)

    
    R0 = si.rol(pt1a,8)&si.rol(pt1a,1)^si.rol(pt1a,2)^pt0a
    R1 = si.rol(pt1b,8)&si.rol(pt1b,1)^si.rol(pt1b,2)^pt0b

    X = si.convert_to_binary([pt0a, pt1a, pt0b, pt1b,R0^R1])
    X = X.reshape(n*n,use_n,pairs,16*5)
    X = X.reshape(n*n*use_n,pairs*16*5)
    
    
    # Z = net.predict(X, batch_size=10000) 
    Z = net.predict(X, batch_size=18000) 
    # 统计得分
    Z = Z/(1-Z+1e-5)
    Z = np.log2(Z)
    Z = Z.reshape(-1, use_n)
    # 获取均值
    v = np.mean(Z, axis=1) * use_n
    # 均值最大值位置
    m = np.argmax(v)
    # 均值最大值
    val = v[m]
    key1 = keys1[m]
    key2 = keys2[m]
    return(key1, key2, val)



# here, we use some symmetries of the wrong key performance profile
# by performing the optimization step only on the 14 lowest bits and randomizing the others
# on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
# In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here
tmp_br = np.arange(2**14, dtype=np.uint16)
# 重复32次，并修改形状为（2^14,32）
tmp_br = np.repeat(tmp_br, 32).reshape(-1, 32)
# print("tmp_br.shape = ",tmp_br.shape)


def bayesian_rank_kr(cand, emp_mean, m=m11, s=s11):
    # print("bayesian rank kr......")
    global tmp_br
    n = len(cand)
    if (tmp_br.shape[1] != n):
        tmp_br = np.arange(2**14, dtype=np.uint16)
        tmp_br = np.repeat(tmp_br, n).reshape(-1, n)
    tmp = tmp_br ^ cand
    v = (emp_mean - m[tmp]) * s[tmp]
    v = v.reshape(-1, n)
    scores = np.linalg.norm(v, axis=1)
    return(scores)


def bayesian_key_recovery(pairs,cts, net=net12, m=m12, s=s12, num_cand=32, num_iter=5, seed=None):
    # print("bayesian key recovery......")
    # num_cipher = 2**neutral_bits
    # ct shape =(pairs,2**neutral_bits)
    num_cipher = len(cts[0][0])
    # print("num_cipeher = ",num_cipher)
    # print("len(pairs)=%d,len(num_cipher)=%d" % (pairs, num_cipher))
    keys = np.random.choice(2**(WORD_SIZE-2), num_cand, replace=False)
    scores = 0
    best = 0
    if (not seed is None):
        keys = np.copy(seed)
    # cts[] 原始shape为 (pairs,num_cipher),进行flatten操作，shape变为(pairs*num_cipher),
    # 进行tile操作shape变为pairs*num_cipher*num_cand
    # shape变成了（num_cipher,pairs）
    cts0=cts[0].transpose().flatten()
    cts1=cts[1].transpose().flatten()
    cts2=cts[2].transpose().flatten()
    cts3=cts[3].transpose().flatten()

    ct0a, ct1a, ct0b, ct1b = np.tile(cts0, num_cand), np.tile(
        cts1, num_cand), np.tile(cts2, num_cand), np.tile(cts3, num_cand)

    n = pairs*num_cipher

    scores = np.zeros(2**(WORD_SIZE-2))
    used = np.zeros(2**(WORD_SIZE-2))
    # 32*5=160个密钥
    all_keys = np.zeros(num_cand * num_iter, dtype=np.uint16)
    all_v = np.zeros(num_cand * num_iter)
    for i in range(num_iter):

        k = np.repeat(keys, n)
        c0a, c1a = si.dec_one_round((ct0a, ct1a), k)
        c0b, c1b = si.dec_one_round((ct0b, ct1b), k)

        R0 = si.rol(c1a,8)&si.rol(c1a,1)^si.rol(c1a,2)^c0a
        R1 = si.rol(c1b,8)&si.rol(c1b,1)^si.rol(c1b,2)^c0b
        X  = si.convert_to_binary([c0a, c1a, c0b, c1b,R0^R1])

        X = X.reshape(num_cand,num_cipher,pairs*16*5)
        X = X.reshape(num_cand*num_cipher,pairs*16*5)

        # Z = net.predict(X, batch_size=10000)  
        Z = net.predict(X, batch_size=18000) 
        Z = Z.reshape(num_cand, -1)
        # 对行求均值
        means = np.mean(Z, axis=1)
        Z = Z/(1-Z+1e-5)
        Z = np.log2(Z)
        v = np.sum(Z, axis=1)
        all_v[i * num_cand:(i+1)*num_cand] = v
        all_keys[i * num_cand:(i+1)*num_cand] = np.copy(keys)
        scores = bayesian_rank_kr(keys, means, m=m, s=s)
        # 找到当前效果最好的密钥
        # tmp = np.argpartition(scores+used, num_cand)
        tmp = np.argpartition(scores, num_cand)
        # 重置密钥
        keys = tmp[0:num_cand]
        r = np.random.randint(0, 4, num_cand, dtype=np.uint16)
        r = r << 14
        keys = keys ^ r
    return(all_keys, scores, all_v)


def test_bayes(device,cts,pairs,it=1, cutoff1=10, cutoff2=10, net=net12, net_help=net11, m_main=m12, m_help=m11, s_main=s12, s_help=s11, verify_breadth=None):
    # ct[0] shape (pairs*num_structure, 2**neutral_bits)
    # print("test bayes......")
    n = len(cts[0])
    # 因为每个数据有pairs个密文
    n = int(n/pairs)
    num_cipher = len(cts[0][0])
    # shape为(pairs*num_structure,2**neutral_bits)
    # verify_breadth =  2**neutral_bits
    if (verify_breadth is None):
        verify_breadth = len(cts[0][0])
    alpha = sqrt(n);
    # alpha = n;
    best_val = -200.0;best_key = (0, 0)
    # 密文结构
    best_pod = 0
    keys = np.random.choice(2**WORD_SIZE, 32, replace=False)
    eps = 0.001
    # 第i个密文结构最高的区分器得分
    local_best = np.full(n, -10)
    # 第i个密文结构迭代的次数
    num_visits = np.full(n, eps)

    for j in range(it):
        # if j%2000 ==0:
        # print("the "+str(j)+"th iter")
        # print("best_val = ",best_val)
        # upper confidence bounds 公式
        # local_best  第i个密文结构最高的区分器得分 num_visits 第i个密文结构之前的迭代次数
        priority = local_best + alpha * np.sqrt(log2(j+1) / num_visits)
        # 优先测试第i个密文结构
        i = np.argmax(priority)
        # print("cipher structure = ",i)
        num_visits[i] = num_visits[i] + 1
        if (best_val > cutoff2):
            improvement = (verify_breadth > 0)
            while improvement:
                # best_pod为不同的密文结构
                # 到了最后阶段，验证猜测的密钥是否正确
                print("cutoff the device " +str(device) + " best_pod = "+str(best_pod))
                k1, k2, val = verifier_search(pairs,[cts[0][best_pod::n], cts[1][best_pod::n], cts[2][best_pod::n], cts[3][best_pod::n]], best_key, net=net_help, use_n=verify_breadth)
                # print("val = ", val)
                improvement = (val > best_val)
                if (improvement):
                    best_key = (k1, k2);best_val = val;
            return(best_key, j)
        # keys shape = (num_cand*num_iter)
        keys, scores, v = bayesian_key_recovery(
            pairs,[cts[0][i::n], cts[1][i::n], cts[2][i::n], cts[3][i::n]], num_cand=32, num_iter=5, net=net, m=m_main, s=s_main)
        vtmp = np.max(v)
        # if vtmp > 0:
        #     print("vtmp = ",vtmp)
        print("vtmp = ",vtmp)
        if (vtmp > local_best[i]):
            local_best[i] = vtmp
        if (vtmp > cutoff1):
            # print("in 2 round")
            
            l2 = [i for i in range(len(keys)) if v[i] > cutoff1]
            
            for i2 in l2:
                # print("first key %x" % keys[i2])
                # print("vtmp = ",vtmp)
                c0a, c1a = si.dec_one_round((cts[0][i::n], cts[1][i::n]), keys[i2])
                c0b, c1b = si.dec_one_round((cts[2][i::n], cts[3][i::n]), keys[i2])

                keys2, scores2, v2 = bayesian_key_recovery(
                    pairs,[c0a, c1a, c0b, c1b], num_cand=32, num_iter=5, m=m_help, s=s_help, net=net_help)
                vtmp2 = np.max(v2)
                # print("second key %x" % keys2[np.argmax(v2)])
                # if vtmp2 > -200:
                print("vtmp2 = ",vtmp2)
                if (vtmp2 > best_val):
                    best_val = vtmp2
                    best_key = (keys[i2], keys2[np.argmax(v2)])
                    best_pod = i
    improvement = (verify_breadth > 0)
    while improvement:
        print("the device " +str(device) + " best_pod = "+str(best_pod))
        k1, k2, val = verifier_search(pairs,[cts[0][best_pod::n], cts[1][best_pod::n], cts[2]
                                      [best_pod::n], cts[3][best_pod::n]], best_key, net=net_help, use_n=verify_breadth)
        # print("val = ", val)
        improvement = (val > best_val)
        if (improvement):
            best_key = (k1, k2)
            best_val = val
    return(best_key, it)

def GET1(a,i):
    return ((a & (1<<i)) >> i)

def replace_1bit(a,b,i):

    mask = 0xffff ^ (1<<i)
    a = a & mask
    a = a | (b << i)
    return a 

def test(device,n=3, pairs=8,nr=17, num_structures=2**8, it=2**9, cutoff1=50, cutoff2=2500, neutral_bits=[[2],[3],[4],[6],[8],[9],[10],[18],[22],[0,24],[12,26]], net=net12, net_help=net11, m_main=m12, s_main=s12,  m_help=m11, s_help=s11, verify_breadth=None):
    #         轮数  结构数                迭代轮数    c1界          c2界             中立bit                        密钥扩展算法      主区分器    辅助区分器
  
    # n是要测试的次数
    gpus = tf.config.list_physical_devices(device_type = 'GPU')
    for gpu in gpus:
	    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(devices=gpus[device],device_type='GPU')
    model_wdir = './student_distinguisher_knowledge_distillation/'

    net = load_model(model_wdir+"simon_model_12r_num_epochs10_acc_0.5086619853973389.h5")
    net_help = load_model(model_wdir+"simon_model_11r_num_epochs40_acc_0.558271.h5")

    arr1 = np.zeros(n, dtype=np.uint16)
    arr2 = np.zeros(n, dtype=np.uint16)
    # 记录开始时间
    t0 = time()
    data = 0
    # print("cutoff1 = %d cuttoff2 = %d" %(cutoff1,cutoff2))
    # 测试n次
    for i in range(n):
        print("Test:", i)
        # 使用相同密钥加密的密文
        key = gen_key(nr)
        
        pt0,pt1 = gen_plain(num_structures)

        set_bit_x1 = GET1(key[0],1)
        pt0 = replace_1bit(pt0,set_bit_x1,1)
        pt1 = replace_1bit(pt1,set_bit_x1,1)

        set_bit_x3 = GET1(key[0],3)
        pt0 = replace_1bit(pt0,set_bit_x3,3)
        pt1 = replace_1bit(pt1,set_bit_x3,3)

        td = test_correct_pairs(pt0, pt1, key)
        # print("td shape",td.shape)
        g = np.sum(td)
        if g == 0:
            # print("dont have")
            arr1[i]=0xffff
            arr2[i]=0xffff
            continue
        else :
            print("the device "+str(device)+" td == 1 indice "+str(np.where(td==1)))
        
        pt0 = pt0[np.where(td==1)]
        pt1 = pt1[np.where(td==1)]
        
        pt0,pt1,_,_ = make_structure(pt0,pt1,neutral_bits=[[2],[3],[4]])

        pt0 = pt0.transpose().flatten()
        pt1 = pt1.transpose().flatten()

        ct = gen_challenge(
            pt0,pt1,key, neutral_bits=neutral_bits)
        # 没有将nr传递到find_good,所以find_good函数采用的是自己的默认值
        # print("ct[0] shape",ct[0].shape)
        # print("true_key  %x %x" % (key[nr-1],key[nr-2]))
        
        guess, num_used = test_bayes(device,ct,pairs=pairs,it=it, cutoff1=cutoff1, cutoff2=cutoff2, net=net, net_help=net_help,
                                     m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help, verify_breadth=verify_breadth)
        
        # print("num_used = ", num_used)
        # 这里算的是数据复杂度，所以要取最小值，num_used的值可能会超过100，但数据结构数最多是100.
        num_used = min(num_structures, num_used)
    
        # 这里有个2*应该除以1/2，成功地概率大于50%,其实这里最合适应该写成功概率
        data = data + 2 * (2 ** len(neutral_bits)) * num_used
        # 两轮密钥猜测结果和真实结果对比
        # 因为key[nr-1]存储了pairs个相同的密钥
        arr1[i] = guess[0] ^ key[nr-1]
        arr2[i] = guess[1] ^ key[nr-2]
        # print("guess 1 round key %x" %guess[0])
        print("Difference between real key and key guess: ",
              hex(arr1[i]), hex(arr2[i]))
    t1 = time()
    print("Done.")
    d1 = [hex(x) for x in arr1]
    d2 = [hex(x) for x in arr2]
    print("Differences between guessed and last key:", d1)
    print("Differences between guessed and second-to-last key:", d2)
    print("Wall time per attack (average in seconds):", (t1 - t0)/n)
    # 对数除法变加法
    print("Data blocks used (average, log2): ", log2(data) - log2(n))
    # return(arr1, arr2, good)
    return(arr1, arr2)




if __name__ == "__main__":


    
    neutral_bits = [[2],[3],[4],[6],[8],[9],[10],[18],[22],[0,24],[12,26]]
  
    success_rate = []
    nr = 17

    PN = 5
    pool = mp.Pool(PN)
    idx_range= range(0,PN)
    # results = pool.map_async(test,idx_range).get()
    results = pool.starmap(test,[(device,) for device in idx_range])
    results = np.array(results)

    arr1 = []
    arr2 = []
    for result in results:
        for arr in result[0]:
            arr1.append(arr)
        for arr in result[1]:
            arr2.append(arr)
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    print("cutoff_success rate",np.sum(arr1==0)/len(arr1))
    success_rate.append(np.sum(arr1==0)/len(arr1))
    np.save(open(str(nr)+'round_run_sols.npy', 'wb'), [arr1,arr2])