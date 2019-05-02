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
def get_boxes(nbr_boxes, pc):

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

    x_max = np.floor(Nx)*box_length + x_min + box_length/2; Nx = np.floor(Nx)
    y_max = np.floor(Ny)*box_length + y_min + box_length/2; Ny = np.floor(Ny)
    z_max = np.floor(Nz)*box_length + z_min + box_length/2; Nz = np.floor(Nz)

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

    boxes = []
    total = len(xyz)
    cnt = 0
    for index in range(len(xyz)):

        cnt+=1
        sys.stdout.write("Getting boxes {}/{}...\r".format(cnt, nbr_boxes))
        sys.stdout.flush()
    
        x_min = xyz[index, 0]; x_max = x_min + box_length
        y_min = xyz[index, 1]; y_max = y_min + box_length
        z_min = xyz[index, 2]; z_max = z_min + box_length

     
        x_number = np.where(x_axis == round(x_min,4))[0][0]
        y_number = np.where(y_axis == round(y_min,4))[0][0]
        z_number = np.where(z_axis == round(z_min,4))[0][0]
     
        indices = ((pc[:,0] >= x_min) & (pc[:,0] < x_max) & (pc[:,1] >= y_min) & (pc[:,1] < y_max) & (pc[:,2] >= z_min) & (pc[:,2] < z_max))


        points_in_box = pc[indices]
        #Dont save box if there are no points
        if len(points_in_box) == 0:
            continue
            
        box_name = "Box_" + str(x_number) + "_" + str(y_number) + "_" + str(z_number)

        # Add the list indices to the boxes list if list is not empty, and the box name
        boxes.append([points_in_box, box_name, {"x":[x_min,x_max] ,"y":[y_min,y_max] , "z": [z_min,z_max]} ] )

        

    print("")
    return boxes

if __name__ == '__main__':

    import random
    from utils import make_plot

    nbr_boxes = 10000
    plot_data = np.load("Data/ply_2cams_forOli_1.npy") #
    # plot_points_used= np.random.rand(100000,3)
    # labels_used= np.random.rand(100000, 3)
    plot_points_used = plot_data[0]
    labels_used = plot_data[1]
    make_plot(plot_points_used, labels_used)
    t1=clock()
    list_box = get_boxes(nbr_boxes, plot_points_used, labels_used)
    t2=clock()
    time = t2-t1
    print("time = ", time, "sec")

    total=0
    total_boxes = 0
    for box in list_box:
        num_points= len(box[0])
        box_name = box[2]
        # print(box_name)
        # print(box[0])
        total+=num_points
        total_boxes +=1

        if total_boxes == 1:
            
            plot_points = box[0]
            labels = box[1]

        elif total_boxes < 800:
            plot_points = np.concatenate((plot_points, box[0]), axis = 0)
            labels = np.concatenate((labels, box[1]), axis = 0)

    # print(plot_points.shape, labels.shape)
    make_plot(plot_points, labels)
    print("total points", total)
    print("N boxes =",total_boxes)