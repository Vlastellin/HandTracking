import pickle
import numpy as np
import matplotlib.pyplot as plt
with open('data_pair.pickle', 'rb') as f:
     points_list, y_list =pickle.load( f)
color=['0.0','0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9']
#print(y_list)
for i,point in enumerate(points_list):
    #print(y_list[i])
    plt.scatter(point[0], point[1], c=[color[y_list[i]]])
plt.show()