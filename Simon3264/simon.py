
#L_{i+1}:=((L_{i}>>alpha)+(mod) R_{i})+k, R_{i+1}:=(R_{i} << beta)+L_{i+1}
import numpy as np
from os import urandom
from collections import deque

def WORD_SIZE():
    return(16);

MASK_VAL = 2 ** WORD_SIZE() - 1;
def rol(x,s):
    return(((x << s) & MASK_VAL) | (x >> (WORD_SIZE() - s)));

def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    # print("c0 shape",c0)
    ls_1_x = ((c0 >> (WORD_SIZE() - 1)) + (c0 << 1)) & MASK_VAL
    ls_8_x = ((c0 >> (WORD_SIZE() - 8)) + (c0 << 8)) & MASK_VAL
    ls_2_x = ((c0 >> (WORD_SIZE() - 2)) + (c0 << 2)) & MASK_VAL

    # XOR Chain
    xor_1 = (ls_1_x & ls_8_x) ^ c1
    xor_2 = xor_1 ^ ls_2_x
    # print("xor_2 = ",xor_2)
    new_c0 = k ^ xor_2
    return (new_c0, c0)

def dec_one_round(c,k):  

    c0, c1 = c[0], c[1];
    ls_1_c1 = ((c1 >> (WORD_SIZE() - 1)) + (c1 << 1)) & MASK_VAL
    ls_8_c1 = ((c1 >> (WORD_SIZE() - 8)) + (c1 << 8)) & MASK_VAL
    ls_2_c1 = ((c1 >> (WORD_SIZE() - 2)) + (c1 << 2)) & MASK_VAL

    # Inverse XOR Chain
    xor_1 = k ^ c0
    xor_2 = xor_1 ^ ls_2_c1
    new_c0 = (ls_1_c1 & ls_8_c1) ^ xor_2
    
    return (c1, new_c0)


def expand_key(k, nr):

    # k = k & ((2 ** 64) - 1)
    zseq = 0b01100111000011010100100010111110110011100001101010010001011111
    ks = []
    k = np.transpose(k)
    for i in range(len(k)):
        ks.append([])
    # k_init = [[((k >> (WORD_SIZE() * (3 - i))) & MASK_VAL) for i in range(4)]]
    # print("k_init = ", k_init)
    for i in range(len(k)):
        k_init = k[i] & MASK_VAL
        k_reg = deque(k_init) 
        rc = MASK_VAL ^ 3
        for x in range(nr):
            rs_3 = ((k_reg[0] << (WORD_SIZE() - 3)) + (k_reg[0] >> 3)) & MASK_VAL
            rs_3 = rs_3 ^ k_reg[2]
            rs_1 = ((rs_3 << (WORD_SIZE() - 1)) + (rs_3 >> 1)) & MASK_VAL
            c_z = ((zseq >> (x % 62)) & 1) ^ rc
            new_k = c_z ^ rs_1 ^ rs_3 ^ k_reg[3]
            ks[i].append(k_reg.pop())
            k_reg.appendleft(new_k)
    ks = np.array(ks,dtype=np.uint16)
    ks = np.transpose(ks)

    return ks

def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        # print("k  shape",k.shape)
        # print("k shape",k.shape)
        x,y = enc_one_round((x,y), k);
    return(x, y);

def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

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


def make_train_data(n, nr, pairs,diff=(0x0000,0x0040)):
    
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    Y1= np.tile(Y,pairs);
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    keys = np.tile(keys,pairs);
    plain0l = np.frombuffer(urandom(2*n*pairs),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n*pairs),dtype=np.uint16);
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    num_rand_samples = np.sum(Y1==0);

    plain1l[Y1==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    plain1r[Y1==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    ks = expand_key(keys, nr);
    # print("ks = ",ks)

    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);

    R0 = rol(ctdata0r,8)&rol(ctdata0r,1)^rol(ctdata0r,2)^ctdata0l
    R1 = rol(ctdata1r,8)&rol(ctdata1r,1)^rol(ctdata1r,2)^ctdata1l

    
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r,R0^R1]);
    
    X = X.reshape(pairs,n,16*5).transpose((1,0,2))
    X = X.reshape(n,1,-1)
    X = np.squeeze(X)
    # print("******")
    return (X,Y);

#real differences data generator
# def real_differences_data(n, nr, pairs=2, diff=(0x0000,0x0040)):
#     #generate labels
#     Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
#     Y1= np.tile(Y,pairs);
#     #generate keys
#     keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    
#     keys = np.tile(keys,pairs);
#     #generate plaintexts
#     plain0l = np.frombuffer(urandom(2*n*pairs),dtype=np.uint16);
#     plain0r = np.frombuffer(urandom(2*n*pairs),dtype=np.uint16);
#     #apply input difference
#     plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
#     num_rand_samples = np.sum(Y1==0);
#     #expand keys and encrypt
#     ks = expand_key(keys, nr);
#     ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
#     ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
#     #generate blinding values
#     #加入噪声
#     k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#     k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
#     #apply blinding to the samples labelled as random
#     ctdata0l[Y1==0] = ctdata0l[Y1==0] ^ k0; ctdata0r[Y1==0] = ctdata0r[Y1==0] ^ k1;
#     ctdata1l[Y1==0] = ctdata1l[Y1==0] ^ k0; ctdata1r[Y1==0] = ctdata1r[Y1==0] ^ k1;
#     #convert to input data for neural networks
#     R0 = ror(ctdata0l^ctdata0r,BETA())
#     R1 = ror(ctdata1l^ctdata1r,BETA())
#     X = convert_to_binary([R0,R1,ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
#     X = X.reshape(pairs,n,16*6).transpose((1,0,2))
#     X = X.reshape(n,1,-1)
#     X = np.squeeze(X)
#     return(X,Y);

if __name__ == "__main__":
    num_rounds = 4
    X, Y = make_train_data(1,num_rounds);
    # print(X)
    # print(Y)

    # plain0l = 0x6565;plain0r = 0x6877
    # keys = np.array([0x1918,0x1110,0x0908,0x0100],dtype=np.uint16)
    # keys = np.array([keys],dtype=np.uint16)
    # ks = expand_key(keys, 4);
    # ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    # print(hex(ctdata0l),hex(ctdata0l))