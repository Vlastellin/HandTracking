import pickle
import math
import numpy as np
import matplotlib.pyplot as plt


def angle(points, k_point1,k_point2,k_point3):
    a=math.sqrt(math.pow(points[k_point1][0]-points[k_point2][0],2)+math.pow(points[k_point1][1]-points[k_point2][1],2))
    b=math.sqrt(math.pow(points[k_point3][0]-points[k_point2][0],2)+math.pow(points[k_point3][1]-points[k_point2][1],2))
    c=math.sqrt(math.pow(points[k_point3][0]-points[k_point1][0],2)+math.pow(points[k_point3][1]-points[k_point1][1],2))
    return (np.arccos((a*a+b*b-c*c)/(2*a*b)))*180/np.pi
 

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



if __name__ == '__main__':
    
    #loading a list of sets of 2 d coordinates of hand keypoints
    with open('data.pickle', 'rb') as f:
        points_list =pickle.load( f)
    marks = generate_marks()
    points_list, marks = delete_zeros_record(points_list, marks)
       
    color=['0.0','0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9']
        
    for i,points in enumerate(points_list):
        if marks[i] in [0,1,4,7,8]:
            y_kord=i*5
            x_kord=angle(points,5,0,17)
            plt.scatter(y_kord,x_kord, c=[color[2]])        
        
    plt.show()
    
    