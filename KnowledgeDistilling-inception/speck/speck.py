
#L_{i+1}:=((L_{i}>>alpha)+(mod) R_{i})+k, R_{i+1}:=(R_{i} << beta)+L_{i+1}
import numpy as np
from os import urandom
import tensorflow as tf

def WORD_SIZE():
    return(16);

def ALPHA():
    return(7);

def BETA():
    return(2);

#二进制全为1
MASK_VAL = 2 ** WORD_SIZE() - 1;

def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);
#左移
def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));
#右移
def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));



def dec_one_round(c,k):
    c0, c1 = c[0], c[1];
    # print("c[0] shape",c[0].shape)
    # print("c[1] shape",c[1].shape)
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    # print(c1.shape)
    return(c0, c1);

def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i);
    return(ks);

def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    # print(c0.shape)
    c0 = ror(c0, ALPHA());
    #& MASK_VAL 模加操作
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return(c0,c1);

def encrypt(p, ks):
    x, y = p[0], p[1];
    # print(x.shape)
    for k in ks:
        x,y = enc_one_round((x,y), k);
    return(x, y);

def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

def check_testvector():
    key = (0x1918,0x1110,0x0908,0x0100)
    pt = (0x6574, 0x694c)
    ks = expand_key(key, 22)
    print([hex(x) for x in ks])
    ct = encrypt(pt, ks)
    if (ct == (0xa868, 0x42f2)):
        print("Testvector verified.")
        return(True);
    else:
        print("Testvector not verified.")
        return(False);



def convert_to_binary(l):
    n = len(l)
    k = WORD_SIZE() * n
    X = np.zeros((k, len(l[0])), dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - 1 - i % WORD_SIZE()
        X[i] = (l[index] >> offset) & 1
    X = X.transpose()
    return(X)

def make_train_data(n, nr, pairs=2, diff=(0x0040,0)):
    
    # print("the size of data is ",n)
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    Y1= np.tile(Y,pairs);
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    keys = np.tile(keys,pairs);
    plain0l = np.frombuffer(urandom(2*n*pairs),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n*pairs),dtype=np.uint16);

    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    num_rand_samples = np.sum(Y1==0);

    plain1l[Y1==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    plain1r[Y1==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    R0 = ror(ctdata0l^ctdata0r,BETA())
    R1 = ror(ctdata1l^ctdata1r,BETA())
    X = convert_to_binary([R0,R1,ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    
    X = X.reshape(pairs,n,16*6).transpose((1,0,2))
    X = X.reshape(n,1,-1)
    X = np.squeeze(X)
    # Y = tf.one_hot(indices=Y, depth=2)
    return(X,Y);

def make_train_data_gohr(n, nr, pairs=2, diff=(0x0040,0)):
    
    # print("the size of data is ",n)
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    Y1= np.tile(Y,pairs);
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    keys = np.tile(keys,pairs);
    plain0l = np.frombuffer(urandom(2*n*pairs),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n*pairs),dtype=np.uint16);

    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    num_rand_samples = np.sum(Y1==0);

    plain1l[Y1==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    plain1r[Y1==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    R0 = ror(ctdata0l^ctdata0r,BETA())
    R1 = ror(ctdata1l^ctdata1r,BETA())
    X = convert_to_binary([R0,R1,ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    
    X = X.reshape(pairs,n,16*6).transpose((1,0,2))
    X = X.reshape(n,1,-1)
    X = np.squeeze(X)
    return(X,Y);

if __name__ == "__main__":
    # num_rounds = 5
    # X, Y = make_train_data(4,num_rounds);
    check_testvector()