import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
        
def generate_marks():
    y_temp=[]
    for i in range(9):
        for j in range(61):
            y_temp.append(i)
    return y_temp


def delete_zeros_record(points_list, marks):
    temp_points_list=[]
    temp_marks=[]
    i=-1
    for points_list_obj in points_list:
        i+=1
        temp_l=[]
        f=True
        points_list_obj.pop()
        for p in points_list_obj:
            if not p==None:
                temp_l.append(p[0])
                temp_l.append(p[1])
            else:
                f=False
                break
        if f:
            #print(temp_l)
            temp_points_list.append(temp_l)
            temp_marks.append(marks[i])
    return temp_points_list, temp_marks
    



def save_points_array_2d(points_array_2d, marks):
    with open('data_points_array_2d.pickle', 'wb') as f:
        pickle.dump([points_array_2d, marks], f)
        
def import_points_array_2d():
    with open('data_points_array_2d.pickle', 'rb') as f:
        points_array_2d, marks =pickle.load( f)
    return points_array_2d, marks
    





if __name__ == '__main__':
    #loading a list of sets of 2 d coordinates of hand keypoints
    with open('data.pickle', 'rb') as f:
        points_list =pickle.load( f)
    marks = generate_marks()
    points_list, marks = delete_zeros_record(points_list, marks)
    points_array= np.array(points_list)
    print(points_array)
    points_array_2d = TSNE(n_components=2).fit_transform(points_array)
    save_points_array_2d(points_array_2d, marks)
    #points_array_2d, marks=import_points_array_2d()
    color=['0.0','0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9']
    
    for i,point in enumerate(points_array_2d):
        plt.scatter(point[0], point[1], c=[color[marks[i]]])
    plt.show()