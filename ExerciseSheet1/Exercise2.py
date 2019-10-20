import numpy as np
import matplotlib.pyplot as plt
plt.rc("text", usetex=True)
##Exercise 2

##2a) Kernel implementation

def K(x,mu=0, w=1):
    return np.where(np.logical_and((x-mu)/w<1, (x-mu)/w>-1) , 15/16*(1-((x-mu)/w)**2)**2, 0)

x=np.linspace(-1,1,100)
plt.plot(x,K(x))
plt.show()

##2b) Kernel density Estimation

example_samples=np.load("samples.npy")[:50]
def KDE(x, w, samples):
    return 1/len(samples)*np.array(sum([K(x,s,w) for s in samples]))
x=np.linspace(-10,20,1000)
for w in [0.1,0.5,1,3.5]:
    plt.plot(x, KDE(x,w, example_samples), label="$\omega = "+str(w)+"$")
plt.legend()
plt.show()
