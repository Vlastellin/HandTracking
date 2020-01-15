import numpy as np
import math

def find_plane_parametrs(coord):
    """
    Finds the parameters (a,b,c) of the plane equation(ax+by+cz=1) from the coordinates of three points
    """
    matrix = np.array([[coord[0][0], coord[0][1], 1.], [coord[1][0], coord[1][1], 1.], [coord[2][0], coord[2][1], 1.]]) 
    vector = np.array([coord[0][2],coord[1][2], coord[2][2]]) 
    return np.linalg.solve(matrix, vector)

def angle(plane1, plane2):
    """
    Finds the angle between two planes
    """
    A1, B1, C1 = plane1[0], plane1[1],  plane1[2]
    A2, B2, C2 = plane2[0], plane2[1],  plane2[2]
    cos = math.fabs(A1*A2+B1*B2+C1*C2)/(math.sqrt(A1*A1+B1*B1+C1*C1)*math.sqrt(A2*A2+B2*B2+C2*C2))
    i=1
    if (A1*A2+B1*B2+C1*C2)<0:
        i=-1
    return i*math.degrees(math.acos(cos))


def description_of_hand_position(keypoint_coord3d_v):
    """
    Finds the angle between the plane and the three coordinate axes
    Input:
        keypoint_coord3d_v:  [1, 21, 3] tf.float32 tensor, Normalizable 3D coordinates of  keypoints
    Output:
        result_str: string, line describing the angles between the plane and the three coordinate axes    
    """
    coord_for_plotting_plane=[[keypoint_coord3d_v[0][0][0],keypoint_coord3d_v[0][0][1],keypoint_coord3d_v[0][0][2] ],[keypoint_coord3d_v[0][8][0], keypoint_coord3d_v[0][8][1],keypoint_coord3d_v[0][8][2] ],[keypoint_coord3d_v[0][20][0], keypoint_coord3d_v[0][20][1],keypoint_coord3d_v[0][20][2] ]]
    plane_parametrs = find_plane_parametrs(coord_for_plotting_plane)
    plane=[plane_parametrs[0],plane_parametrs[1],1] 
    angle_with_oy=int(angle(plane, [0,1,0]))
    angle_with_ox=int(angle(plane, [1,0,0]))
    angle_with_oz=int(angle(plane, [0,0,1]))
    result_str="углы с осями (в градусах): с OX "+str(angle_with_ox)+", " 
    result_str+="с OY "+str(angle_with_oy)+", " 
    result_str+="с OZ "+str(angle_with_oz)     
    return result_str