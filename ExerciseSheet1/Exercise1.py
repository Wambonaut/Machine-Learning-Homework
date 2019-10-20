import numpy as np
import matplotlib.pyplot as plt


##Exercise 1a: Principal Component Analysis
protein_data=np.loadtxt("protein.txt")
print(protein_data)
principal_component_matrix=np.matmul(np.transpose(protein_data),protein_data)#create the XtX matrix
eigenv=np.linalg.eigh(principal_component_matrix)
print(eigenv[1][0])

#calculate the 2 most important coordinates of the points in the pricipal-component-system
x=np.matmul(protein_data,eigenv[1][:,-1])
y=np.matmul(protein_data,eigenv[1][:,-2])

#make a scatter plot
plt.scatter(x, y)
countries=open("countries.txt","r")

##annotate the scatter plot)
for i,txt in enumerate(countries):
    print(txt)
    plt.annotate(txt, (x[i],y[i]))


plt.show()
