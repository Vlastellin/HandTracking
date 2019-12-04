import pickle
import math
def angle(points, ind1,ind2,ind3):
    a=math.sqrt(math.pow(points[ind1][0]-points[ind2][0],2)+math.pow(points[ind1][1]-points[ind2][1],2))
    b=math.sqrt(math.pow(points[ind3][0]-points[ind2][0],2)+math.pow(points[ind3][1]-points[ind2][1],2))
    c=math.sqrt(math.pow(points[ind3][0]-points[ind1][0],2)+math.pow(points[ind3][1]-points[ind1][1],2))
    return (np.arccos((a*a+b*b-c*c)/(2*a*b)))*180/np.pi
 
with open('data.pickle', 'rb') as f:
     points_list =pickle.load( f)
        
#print(len(points_list))
y_temp=[]
for i in range(9):
    for j in range(61):
        y_temp.append(i)
y=[]
i=-1
import numpy as np
from sklearn.manifold import TSNE
p_l=[]
for l in points_list:
    i+=1
    temp_l=[]
    f=True
    l.pop()
    for p in l:
        if not p==None:
            temp_l.append(p)
            
            #print(y_temp[i])
        else:
            f=False
            break
    if f:
        #print(temp_l)
        p_l.append(temp_l)
        y.append(y_temp[i])
    

#p_l

color=['0.0','0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9']
print(len(y))
import matplotlib.pyplot as plt
for i,points in enumerate(p_l):
    if y[i] in [0,1,4,7,8]:
        #print(y_list[i])
        y_kord=i*5
        #find points
        #x_kord=angle(points,5,0,9)
        #plt.scatter( y_kord,x_kord, c=[color[0]])
        #x_kord=angle(points,5,0,13)
        #plt.scatter(y_kord,x_kord, c=[color[1]])
        x_kord=angle(points,5,0,17)
        plt.scatter(y_kord,x_kord, c=[color[2]])
        #x_kord=angle(points,9,0,13)
        #plt.scatter(y_kord,x_kord, c=[color[3]])
        #x_kord=angle(points,9,0,17)
        #plt.scatter(y_kord,x_kord, c=[color[4]])
        #x_kord=angle(points,13,0,17)
        #plt.scatter(y_kord,x_kord, c=[color[5]])
plt.show()