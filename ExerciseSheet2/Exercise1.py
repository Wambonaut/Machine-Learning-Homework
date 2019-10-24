import numpy as np
import matplotlib.pyplot as plt


##Exercise 1: Mean-shift and K-Means
##1a) Implement the Epanechikov-Kernel

def k(x,mu=0,w=1):
    return np.where(np.abs((x-mu)/w)<1, 3/(4*w)*(1-((x-mu)/w)**2), 0)
x=np.linspace(-3,3,100)
plt.plot(x, k(x))
plt.show()

##1b) Mean-shift on a 1d data set

def mean_shift_step(data_points):
    for i,x0 in enumerate(data_points):
        data_points[i]=sum(np.where(np.abs(x0-data_points)<1, data_points,  0))/sum(np.where(np.abs(x0-data_points)<1, 1, 0))
    return data_points

data1=np.load("meanshift1d.npy")

def KDE(x, w, samples,K):
    return 1/len(samples)*sum([K(x,s,w) for s in samples])

x=np.linspace(-4,4,100)
y=KDE(x,1,data1,k)
plt.plot(x,y)

data=data1
r=10
for i in range(r):
    plt.scatter(data, np.zeros(len(data))+i/r)
    data=mean_shift_step(data)
plt.show()
