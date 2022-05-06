import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FormatStrFormatter



    
fig, axes = plt.subplots(2,2,figsize=(12,8))
# fig, axes = plt.subplots(2,2)

# plt.figure(figsize=(12,8))
axes[0,0].set_xticks(np.arange(0,65537,16384))
axes[0,1].set_xticks(np.arange(0,65537,16384))
axes[1,0].set_xticks(np.arange(0,65537,16384))
axes[1,1].set_xticks(np.arange(0,65537,16384))

axes[0,0].set_title('$r$ = 9')
axes[0,1].set_title("$r$ = 10")
axes[1,0].set_title("$r$ = 11")
axes[1,1].set_title("$r$ = 12")

axes[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
axes[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
axes[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
axes[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

axes[0,0].set_xlabel("Difference to real key")
axes[0,1].set_xlabel("Difference to real key")
axes[1,0].set_xlabel("Difference to real key")
axes[1,1].set_xlabel("Difference to real key")

axes[0,0].set_ylabel("Mean Response")
axes[0,1].set_ylabel("Mean Response")
axes[1,0].set_ylabel("Mean Response")
axes[1,1].set_ylabel("Mean Response")


axes[0,0].plot(range(2**16), np.load("simon_data_wrong_key_mean_"+str(9)+"r_pairs8.npy"))
axes[0,1].plot(range(2**16), np.load("simon_data_wrong_key_mean_"+str(10)+"r_pairs8.npy"))
axes[1,0].plot(range(2**16), np.load("simon_data_wrong_key_mean_"+str(11)+"r_pairs8.npy"))
axes[1,1].plot(range(2**16), np.load("simon_data_wrong_key_mean_"+str(12)+"r_pairs8.npy"))
plt.subplots_adjust(left=0.14,right=0.95,bottom=0.1,top=0.95,wspace = 0.4,hspace=0.4)
fig.savefig("simon.pdf")

   