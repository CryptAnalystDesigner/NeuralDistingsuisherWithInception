import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FormatStrFormatter

# rounds = [6,7,8]
# pairs = [4,8,16,32]
# for r in rounds:
#     for p in pairs:
#         fig, ax = plt.subplots() 
#         y = np.load("data_wrong_key_mean_"+str(r)+"r_pairs"+str(p)+".npy")
#         ax.plot(range(2**16), y)
#         fig.savefig(str(r)+"-"+str(p)+".png")

rounds = [8]
for r in rounds:

    fig, axes = plt.subplots(3,2)

    # y_major_locator=MultipleLocator(0.2)
    # axes[0,0].yaxis.set_major_locator(y_major_locator)
    # axes[0,1].yaxis.set_major_locator(y_major_locator)
    # axes[1,0].yaxis.set_major_locator(y_major_locator)
    # axes[1,1].yaxis.set_major_locator(y_major_locator)
    # axes[2,0].yaxis.set_major_locator(y_major_locator)
    # axes[2,1].yaxis.set_major_locator(y_major_locator)

    # axes[0,0].set_ylim([0,1])
    # axes[0,1].set_ylim([0,1])
    # axes[1,0].set_ylim([0,1])
    # axes[1,1].set_ylim([0,1])
    # axes[2,0].set_ylim([0,1])
    # axes[2,1].set_ylim([0,1])
    
    # y2 = np.tile(np.array([0.500]),2**16)
    axes[0,0].set_title('$\mathcal{ND}_{bd}$')
    axes[0,1].set_title("$k$ = 2")
    axes[1,0].set_title("$k$ = 4")
    axes[1,1].set_title("$k$ = 8")
    axes[2,0].set_title("$k$ = 16")
    axes[2,1].set_title("$k$ = 32")

    axes[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axes[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axes[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axes[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axes[2,0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axes[2,1].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    axes[0,0].set_xlabel("Difference to real key")
    axes[0,1].set_xlabel("Difference to real key")
    axes[1,0].set_xlabel("Difference to real key")
    axes[1,1].set_xlabel("Difference to real key")
    axes[2,0].set_xlabel("Difference to real key")
    axes[2,1].set_xlabel("Difference to real key")

    axes[0,0].set_ylabel("Mean Response")
    axes[0,1].set_ylabel("Mean Response")
    axes[1,0].set_ylabel("Mean Response")
    axes[1,1].set_ylabel("Mean Response")
    axes[2,0].set_ylabel("Mean Response")
    axes[2,1].set_ylabel("Mean Response")
    
    axes[0,0].plot(range(2**16), np.load("gohr_data_wrong_key_mean_"+str(r)+"r.npy"))
    axes[0,1].plot(range(2**16), np.load("data_wrong_key_mean_"+str(r)+"r_pairs2.npy"))
    axes[1,0].plot(range(2**16), np.load("data_wrong_key_mean_"+str(r)+"r_pairs4.npy"))
    axes[1,1].plot(range(2**16), np.load("data_wrong_key_mean_"+str(r)+"r_pairs8.npy"))
    axes[2,0].plot(range(2**16), np.load("data_wrong_key_mean_"+str(r)+"r_pairs16.npy"))
    axes[2,1].plot(range(2**16), np.load("data_wrong_key_mean_"+str(r)+"r_pairs32.npy"))

    # axes[0,0].plot(range(2**16), y2)
    # axes[0,1].plot(range(2**16), y2)
    # axes[1,0].plot(range(2**16), y2)
    # axes[1,1].plot(range(2**16), y2)
    # axes[2,0].plot(range(2**16), y2)
    # axes[2,1].plot(range(2**16), y2)

    plt.subplots_adjust(left=0.14,right=0.95,bottom=0.1,top=0.95,wspace = 0.6,hspace=0.8)
    fig.savefig(str(r)+" round"".pdf")

   