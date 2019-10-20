import numpy as np
import matplotlib.pyplot as plt

##Exercise 1: Principal Component Analysis

##1a) Data visualization
protein_data=np.loadtxt("protein.txt")
principal_component_matrix=np.matmul(np.transpose(protein_data),protein_data)#create the XtX matrix
eigenv=np.linalg.eigh(principal_component_matrix)

#calculate the 2 most important coordinates of the points in the pricipal-component-system
x=np.matmul(protein_data,eigenv[1][:,-1])
y=np.matmul(protein_data,eigenv[1][:,-2])

#make a scatter plot
countries=open("countries.txt","r")
plt.scatter(x, y)

##annotate the scatter plot)
for i,txt in enumerate(countries):
    print(txt)
    plt.annotate(txt, (x[i],y[i]))


plt.show()
##1b) Lossy Compression

##plot the Eigenvalues
print(eigenv)
plt.plot(eigenv[0][::-1], label="Eigenvalues ordered by size")
plt.plot([sum(eigenv[0][0:-x-1]) for x in range(len(eigenv[0])-1)], label="Error when dropping smaller Eigenvalues")
#plt.yscale("log")
plt.show()
