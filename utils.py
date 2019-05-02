import numpy as np 
import os
import time
import pptk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib, time
import mpl_toolkits.mplot3d.art3d as art3d

matplotlib.interactive(True)



class plot3dClass( object ):

    def __init__( self, points, centers = None):
 
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )
        # Hide grid lines
        self.ax.grid(False)
        max_range = np.array([points[:,0].max()-points[:,0].min(), points[:,1].max()-points[:,1].min(), points[:,2].max()-points[:,2].min()]).max() / 2.0

        mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
        mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
        mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')

        maxPoints = 10000
        if len(points) > maxPoints:

            random_indices= random.sample(range(0,len(points)), maxPoints)
            points = points[random_indices, :]
        self.points = self.ax.scatter(points[:,0], points[:,1], points[:,2], color = [.7,.7,.7, 0.2])
        self.vectors = []
        self.circles = []
        self.bridge_points = []
        self.bridge_point_txt = []
        self.non_branch_points_txt = []
        self.branch_points_txt = []
        self.head_tail_txt = []
        self.connections = []
        self.head_tail = []
        self.non_branch_points = self.ax.scatter(centers[:,0], centers[:,1], centers[:,2], color = [0.8,0,0,1])

        plt.draw() #maybe you want to see this frame?

    def drawCenters(self, myCenters, h):
        
        self.fig.canvas.flush_events()
        for i in range(1):  
            try:
                self.non_branch_points.remove()
            except Exception:
                pass

            try:
                self.vectors.remove()
            except Exception:
                pass
            try:
                for circle in self.circles:
                    circle.remove()
                self.circles = []
            except Exception:
                pass

            try:
                self.bridge_points.remove()
            except Exception:
                pass
            try:
                for connection in self.connections:
                    connection.pop(0).remove()
                self.connections = []
            except Exception:
                pass

            try:
                self.head_tail.remove()
            except Exception:
                pass

        branch_points = []
        non_branch_points = []
        branch_points_txt = []
        non_branch_points_txt = []
        eigen_vectors = []        
        bridge_points = []
        bridge_point_txt = []
        head_tail = []
        head_tail_txt = []
        for center in myCenters:

            if center.label =="branch_point":
                if center.head_tail:
                    head_tail.append(center.center)
                    head_tail_txt.append(center.index)
                else:
                    branch_points.append(center.center)
                    branch_points_txt.append(center.index)

                for connection in center.connections:
                    
                    points = np.array([center.center, myCenters[connection].center])

                    self.connections.append(self.ax.plot(points[:,0],points[:,1], points[:,2],'r-'))
                
            elif center.label =='non_branch_point':
                non_branch_points.append(center.center)
                non_branch_points_txt.append(center.index)

            elif center.label == "bridge_point":
                bridge_point_txt.append(center.index)
                bridge_points.append(center.center)

            vector = tuple(center.center) + tuple(center.eigen_vectors[:,0]/10)
            eigen_vectors.append(vector)

        branch_points = np.array(branch_points)
        bridge_points = np.array(bridge_points)
        non_branch_points = np.array(non_branch_points)
        eigen_vectors = np.array(eigen_vectors)
        head_tail = np.array(head_tail)


        if branch_points.any():
            self.branch_points = self.ax.scatter(branch_points[:,0], branch_points[:,1], branch_points[:,2], color = [0,0.8,0,1])

        if bridge_points.any():
            self.bridge_points = self.ax.scatter(bridge_points[:,0], bridge_points[:,1], bridge_points[:,2], color = [0,0,.8,1])

        if non_branch_points.any():
            self.non_branch_points = self.ax.scatter(non_branch_points[:,0], non_branch_points[:,1], non_branch_points[:,2], color = [0.8,0,0,1])

        if head_tail.any():
            self.head_tail = self.ax.scatter(head_tail[:,0], head_tail[:,1], head_tail[:,2], color = [1,1,0,1])

            for center in non_branch_points[:5]:
                p = Circle((center[0],center[1]), h, fill = False, color =[0.8,0.4,0,0.2])
                self.circles.append(self.ax.add_patch(p))
                art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        #NUMBERS
        # if bridge_point_txt:
        #     if self.bridge_point_txt:
        #         for txt in self.bridge_point_txt:
        #             # txt.pop(0).remove()
        #             txt.remove()
        #         self.bridge_point_txt = []
        #     for i, txt in enumerate(bridge_point_txt):
        #         self.bridge_point_txt.append(self.ax.text(bridge_points[i,0], bridge_points[i,1], bridge_points[i,2], str(txt), None))

        # if branch_points_txt:
        #     if self.branch_points_txt:
        #         for txt in self.branch_points_txt:
        #             # txt.pop(0).remove()
        #             txt.remove()
        #         self.branch_points_txt=[]
        #     for i, txt in enumerate(branch_points_txt):
        #         self.branch_points_txt.append(self.ax.text(branch_points[i,0], branch_points[i,1], branch_points[i,2], str(txt), None))

        # if non_branch_points_txt:
        #     if self.non_branch_points_txt:
        #         for txt in self.non_branch_points_txt:
        #             # txt.pop(0).remove()
        #             txt.remove()
        #         self.non_branch_points_txt= []
        #     for i, txt in enumerate(non_branch_points_txt):
        #         self.non_branch_points_txt.append(self.ax.text(non_branch_points[i,0], non_branch_points[i,1], non_branch_points[i,2], str(txt), None))

        # if head_tail_txt:
        #     if self.head_tail_txt:
        #         for txt in self.head_tail_txt:
        #             # txt.pop(0).remove()
        #             txt.remove()
        #         self.head_tail_txt= []
        #     for i, txt in enumerate(head_tail_txt):
        #         self.head_tail_txt.append(self.ax.text(head_tail[i,0], head_tail[i,1], head_tail[i,2], str(txt), None))


        # X,Y,Z,U,V,W = zip(*eigen_vectors)
        # self.vectors = self.ax.quiver(X,Y,Z,U,V,W)

        plt.draw()                      # redraw the canvas
        
    def keep(self):
        plt.show(block=True)

def get_local_points(points, centers, h, maxLocalPoints =50000):

    #Get local_points points around this center point
    local_indices = []
    for center in centers:

        x,y,z = center

        #1) first get the square around the center
        where_square = ((points[:,0] >= (x - h)) & (points[:, 0] <= (x + h)) & (points[:,1] >= (y - h)) & 
                            (points[:, 1] <= (y + h)) & (points[:,2] >= (z - h)) & (points[:, 2] <= (z + h)))

        square = points[where_square]
        indices_square = np.where(where_square == True)[0]

        # Get points which comply to x^2, y^2, z^2 <= r^2
        square_squared = np.square(square - [x,y,z])
        where_sphere = np.sum(square_squared, axis = 1) <= h**2
        local_sphere_indices = indices_square[where_sphere]
            
        local_indices.append(local_sphere_indices)

    return local_indices

def delete_rows_array(array, indices):
    """
    Deletes indices from an np array
    """

    list_array = list(array)

    print("Shape array:", array.shape, "Deleting indices",len(indices),":", indices)
    for index in sorted(indices, reverse=True):
        del list_array[index]
    array = np.array(list_array)
    print("shape after:",array.shape)
    return array

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def remove_outliers(points, labels = 1, max_std = 3):
    (r,c) = points.shape

    if r == 3:
        X = points[0,:]
        Y = points[1,:]
        Z = points[2,:]
    elif c == 3:
        X = points[:,0]
        Y = points[:,1]
        Z = points[:,2]

    x_mean = np.mean(X)
    x_std = np.std(X)

    y_mean = np.mean(Y)
    y_std = np.std(Y)

    z_mean = np.mean(Z)
    z_std = np.std(Z)

    #Outliers defined as being further then 3 times std from the mean
    x_outliers1 =  np.array([X > x_mean + max_std*x_std], dtype = np.bool)
    x_outliers2 =  np.array([X < x_mean - max_std*x_std], dtype = np.bool)

    y_outliers1 =  np.array([Y > y_mean + max_std*y_std], dtype = np.bool)
    y_outliers2 =  np.array([Y < y_mean - max_std*y_std], dtype = np.bool)

    z_outliers1 =  np.array([Z > z_mean + max_std*z_std], dtype = np.bool)
    z_outliers2 =  np.array([Z < z_mean - max_std*z_std], dtype = np.bool)

    #gets indices where any of the booleans x_outliers1 ... z_outliers2 are True
    indices_to_delete = np.where(np.logical_or.reduce((x_outliers1, x_outliers2, y_outliers1, y_outliers2, z_outliers1, z_outliers2)))

    if c == 3:
        points_out = np.delete(points,indices_to_delete, axis = 0)
    elif r == 3:
        points_out = np.delete(points,indices_to_delete, axis = 1)

    if not isinstance(labels, int):
        labels_out = np.delete(labels, indices_to_delete, axis = 0)

    return points_out, labels_out


def make_plot(plot_points,colors = False, point_size = 0.0005):
    #label colors should be raning from 0 to 1. 
    if isinstance(colors, bool) :
        colors = np.zeros(plot_points.shape)
        colors[:,0] = 1
    v = pptk.viewer(plot_points)
    v.attributes(colors)
    v.set(point_size=point_size)


def draw_vector_lines(vectors, start_point):
    vector_lines = []
    interval = 0.025
    sizes = np.arange(interval,1,interval)


    for vector in vectors:
        for size in sizes:
            point =  -vector * np.array(size)
                           
            point[0] +=start_point[0]
            point[1] +=start_point[1]
            point[2] +=start_point[2]

            vector_lines.append(point)

    return np.array(vector_lines)

def plot_boxes(myDict):
    cg = []
    labels = [[]]

    cnt =0
    for box in myDict: 

        box = myDict[box]

        if not box.merged and box.contains_points and len(box.connections) >0:
            
            vectors = []
            directional_labels = []
            for connection in box.connections:

                connection = myDict[connection]
                
                cg_to_connect = connection.cg

                vector =  box.cg - cg_to_connect
                
                vectors.append(vector)

                directional_label = box.connections[connection.name]
                directional_labels.append(directional_label)

            connection_points = draw_vector_lines(vectors, box.cg)

            label_connection_points = np.zeros(np.shape(connection_points))

            #Colorize the connection based on the connection  value
            index = 0
            label_index = 0
            N_vecs = len(directional_labels)
            vector_size = int(len(connection_points) / N_vecs)
            for directional_label in directional_labels:                    

                #Based on if the directional label is positive or negative we go from max->min or min->max
                min_val = 5; max_val = 250
                step_size = (max_val - min_val) / vector_size

                if sum(directional_label) > 0:
                    linspace = [max_val - int((index+1)*step_size) for index in range(vector_size)]
                else:
                    linspace = [min_val + int((index+1)*step_size) for index in range(vector_size)]

                # print(linspace)
                #Get the nonzero column
                RGB = np.where(directional_label !=0)[0][0] 
                label_connection_points[label_index:label_index+vector_size, RGB] = linspace


                label_index += vector_size
                index +=1                

            label_cg = [255,255,255]
            # label_connection_points[:,0] = 255
            # label_connection_points[:,1] = 255
            # label_connection_points[:,2] = 255
            if cnt ==0:
                labels[0] = label_cg
                if label_connection_points.shape[0] != 0:
                    labels = np.concatenate((labels, label_connection_points), axis=0)   
                cnt+=1
            else:
                
                if label_connection_points.shape[0] != 0:
                    labels = np.concatenate((labels, [label_cg], label_connection_points), axis=0)
                else:
                    labels = np.concatenate((labels, [label_cg]), axis=0)
        

            cg.append(box.cg)
            if connection_points.any():
                cg.extend(connection_points)

    labels = np.array(labels)
    make_plot(cg, labels/255)

    return cg, labels
    
def draw_cube(x,y,z):
    """
    INPUT:
        - x,y,z are lists of length 2 containing: [min, max]
    OUTPUT:
        - POINTS FORMING THE BOUNDARIES OF THE CUBE
    """

    #From 4 corners we can define 4 vcectors at each corner obtaining all 12 vertices:
    boundaries = [x,y,z]
    corner1 = [x[0], y[0], z[0]]
    corner2 = [x[0], y[1], z[1]]
    corner3 = [x[1], y[0], z[1]]
    corner4 = [x[1], y[1], z[0]]

    corners = [corner1,corner2,corner3,corner4]

    cube_points= []
    cnt = 0
    for corner in corners:
        vectors = []
        for index in range(3):
            boundary = boundaries[index]

            column = index

            if corner[column] == boundary[0]:
                vector = np.zeros(3)
                vector[column] = boundary[0] - boundary[1]

            else:
                vector = np.zeros(3)
                vector[column] = boundary[1] - boundary[0]


            vectors.append(vector)

        points = draw_vector_lines(np.array(vectors), corner)

        if cnt>0:
            cube_points = np.concatenate((cube_points, points), axis = 0)
        else:
            cube_points = points
            cnt+=1

    return cube_points

def draw_cubes(boxes):

    first_box = True
    points = []
    for box in boxes:

        boundaries = box[2]
        
        x = boundaries["x"]
        y = boundaries["y"]
        z = boundaries["z"]

        cube_points = draw_cube(x,y,z)

        if first_box:
            points= cube_points
            first_box = False
        else:
            points = np.concatenate((points,cube_points), axis = 0)
    labels = np.ones(points.shape) * 255
    return points, labels

def make_dim_list(myDict, minimum_dim):
    """
    Return list of strings containing box names who have a Vdim >= minimum_dim
    """
    dim_list = []
    for box in myDict:
        Vdim = myDict[box].Vdim

        if Vdim >= minimum_dim:
            dim_list.append(box)
    return dim_list

def find_all_connections(myDict,threshold):
    for key in myDict.keys():
        if myDict[key].contains_points:
            myDict[key].find_connections(threshold)

    # for key in myDict.keys():
    #     if myDict[key].contains_points:
    #         myDict[key].find_enlarged_connections(threshold)

def find_all_Vpairs(myDict, dim_list):

    for box in dim_list:      
        try: 
            myDict[box].find_v_pairs()
        except KeyError:
            pass

def find_all_Epairs(myDict, dim_list):
    for box in dim_list:
        try: 
            myDict[box].find_e_pairs()
        except KeyError:
            pass

def eat_all_Vpairs(myDict, dim_list):
    """
    Returns
        - succes: true if ONE or more Vpairs were eaten
    """
    success =False
    ate_Vpair = False
    # print("Looking for some delicious Vpairs....")
    for box in dim_list:
        
        try:
            ate_Vpair = myDict[box].eat_v_pair()
        except KeyError:
            pass

        if ate_Vpair:
            success =True

    return success

def eat_all_Epairs(myDict, dim_list):
    #Returns true if ONE or more Epairs were eaten
    success =False
    ate_Epair = False
    for box in dim_list:

        try:
            ate_Epair = myDict[box].eat_e_pair()
        except KeyError:
            pass
        if ate_Epair:
            success =True
    return success

def eat_one_epair(myDict, dim_list):
    success = False
    ate_Epair = False
    for box in dim_list:

        try:
            ate_Epair = myDict[box].eat_e_pair()
        except KeyError:
            pass

        if ate_Epair:
            success = True
            break
    return success

def still_vpairs(myDict, dim_list):
    still_vpairs = False
    total = 0
    for box in dim_list:

        try:
            total+= len(myDict[box].Vpairs)
        except KeyError:
            pass

    if total > 0:
        still_vpairs = True
    return still_vpairs, total

def count_boxes_with_points(myDict):
    cnt = 0
    for key in myDict:
        box = myDict[key]

        if box.contains_points:
            cnt+=1

    return cnt

def count_connections(myDict):
    connections = 0
    for key in myDict:
        box = myDict[key]
        connections+=len(box.connections)
    return connections