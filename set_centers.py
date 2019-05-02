import numpy as np 
from time import clock
from itertools import compress
from utils import *
from math import ceil
import sys

# Find the boundaries of the big box, given 1 point cloud
def find_boundaries_box(plot_points):
    x_min = np.min(plot_points[:, 0])
    x_max = np.max(plot_points[:, 0])
    y_min = np.min(plot_points[:, 1])
    y_max = np.max(plot_points[:, 1])
    z_min = np.min(plot_points[:, 2])
    z_max = np.max(plot_points[:, 2])
    return x_min, x_max, y_min, y_max, z_min, z_max


# Gives the coordinates of the points which are in the boxes and the corresponding name of the boxes
def get_centers(nbr_boxes, pc):

    x_min, x_max, y_min, y_max, z_min, z_max = find_boundaries_box(pc)


    #Get the real box length
    x_L = x_max - x_min
    y_L = y_max - y_min
    z_L = z_max - z_min

    #Get the ratio for L in meter to Number of boxes
    volume = x_L*y_L*z_L
    ratio = nbr_boxes/volume
    weight = ratio**(1./3)

    #Get the number of boxes (Now this is a flot, I.E. 12.5642)
    Nx = weight * x_L
    Ny = weight * y_L
    Nz = weight * z_L

    #Increase the length of the sides to fit to an integer number of boxes
    #We choose this one at the moment

    box_length = x_L/Nx

    x_max = max(np.floor(Nx), 1)*box_length + x_min + box_length/2; Nx =  max(np.floor(Nx), 1)
    y_max = max(np.floor(Ny), 1)*box_length + y_min + box_length/2; Ny =  max(np.floor(Ny), 1)
    z_max = max(np.floor(Nz), 1)*box_length + z_min + box_length/2; Nz =  max(np.floor(Nz), 1)

    x_min = x_min - box_length/2
    y_min = y_min - box_length/2
    z_min = z_min - box_length/2

    #Recalculate the length of the sides of the rectangle
    x_L = x_max - x_min
    y_L = y_max - y_min
    z_L = z_max - z_min

    #Now x_L/N_x has changed  so we adjust the box_length
    box_length = x_L/Nx

    #Get the grid mesh with these sides and number of boxes
    x,y,z = np.mgrid[ x_min : x_max + box_length : box_length, y_min : y_max + box_length : box_length, z_min : z_max + box_length : box_length]
    xyz = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    x_axis = np.round(np.mgrid[ x_min : x_max+box_length*2 : box_length], 4)
    y_axis = np.round(np.mgrid[ y_min : y_max+box_length*2 : box_length], 4)
    z_axis = np.round(np.mgrid[ z_min : z_max+box_length*2 : box_length], 4)

    centers = []
    total = len(xyz)
    cnt = 0
    for index in range(len(xyz)):

        
        x_min = xyz[index, 0]; x_max = x_min + box_length
        y_min = xyz[index, 1]; y_max = y_min + box_length
        z_min = xyz[index, 2]; z_max = z_min + box_length

     
        x_number = np.where(x_axis == round(x_min,4))[0][0]
        y_number = np.where(y_axis == round(y_min,4))[0][0]
        z_number = np.where(z_axis == round(z_min,4))[0][0]
     
        indices = ((pc[:,0] >= x_min) & (pc[:,0] < x_max) & (pc[:,1] >= y_min) & (pc[:,1] < y_max) & (pc[:,2] >= z_min) & (pc[:,2] < z_max))


        
        # labels_in_box = labels[indices]

        #Dont save box if there are no points

        if not (indices.any() == True):
            continue
        for i in range(5):        

            index = np.argmax(indices == True)
            indices[index] = False
            if index > 0:
                point_in_box = pc[index]
                
                center = point_in_box
                # Add the list indices to the boxes list if list is not empty, and the box name
                centers.append(center)

                cnt+=1
                sys.stdout.write("Found centers {}/{}...\r".format(cnt, nbr_boxes))
                sys.stdout.flush()

        

    print("")
    return np.array(centers)