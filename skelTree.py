import numpy as np
import random, os
from utils import *
import time , sys
import pickle


"""
Implementation of the SkelTree algorithm from 'Robust skeleton extraction from imperfect point clouds - 2010'
    
    Workings:
    1) Divide the 3D space containing points into N cubes
    2) Each box has 6 adjacent boxes which can be part of a similar pointcloud object. It finds which boxes are connected and labels the connection direction.
    3) Iteratively merges boxes while trying to preserve the local direction of the pointcloud structure restuling in a skeleton of the tree
"""
class myBox:
    """
    The class needs to be defined in the file which defines 'myDict'
    REASON: It refers to and maniuplates the dictionary `dict_name` which is defined in this file. 
            If the class is imported it does not have aceces to this dictionary and is thus not functional
    """

    def calc_cg(self):
        self.cg = np.mean(self.points, axis = 0)

    def __init__(self, dict_name, box_name, points, use_higher_dimensional_boxes = False):
        """
        INPUTS
            dict_name: Name of the parent dictionary containing all the box objects
            box_name: Name of this box object corresponding to Box'xyz'. For example: Box001
            points: The points contained inside this box
            use_higher_dimensional_boxes: Indicate if you want to try connect higher dimenional octtree boxes if no connections are found at the lowest dimensionality
            
        ATTRIBUTES:
            self.dict_name = name of parrent dict
            self.name = name of the box object
            self.points = all the points present in this box. Is empty np.ndarray when no points are present
            self.contains_points = Boolean indicating if there are points yes or no
            self.cg = Center of Gravity of this box. Is 'None' when no points are present
            self.merged = Boolean stating if this box is merged to another box. If False this is parent box. Either having children or not.
            self.parent = the name of the parent. If no parent (i.e. merged == False) parent == None
            self.children = list of children boxes
            self.connections = dictionary of connected boxes with corresponding directional label as np.array; i.e. self.connections = {"Box001": np.arrray([0,0,1]), .... , "Box100":np.arrray([-1,0,0])}
            self.Vdim = the number of connections
            self.Vdir = the sum of the connections directional labels. i.e. [-1,0,1]
        """
        #####################
        #####SOME CHECKS#####
        #####################
        if not box_name.startswith("Box"):
            raise Exception("box_name should be of this format: `Box'xyz'`. So like Box001...")
        if not isinstance(points, np.ndarray):
            raise Exception("Points should be in ndarray Nx3...")
        if points.any():
            if not points.shape[1] ==3:
                raise Exception("Points should be an np.ndarray Nx3...")

        self.dict_name = dict_name
        self.name = box_name 
        self.points = points
        self.use_higher_dimensional_boxes = use_higher_dimensional_boxes

        #If more than 1 points
        if points.any():# and points.shape[0] > 1:
            self.calc_cg()
            self.contains_points = True
        else:
            self.cg = None 
            self.contains_points = False

        self.merged = False
        self.parent = None
        self.children = []
        self.connections = {}
        self.Vdim = 0
        self.Vdir = []
        self.Vpairs = []
        self.Epairs = []

    def merged_with(self, box_name):
        """
        Set everything to merge status, deleting everything
        """
        self.parent = box_name
        self.merged  =True

        self.points = None
        self.points = np.array([])

        self.contains_points = False
        self.cg = None
        
        self.connections = {}
        self.children = []
        self.connections = {}
        self.Vdim = 0
        self.Vdir = []
        self.Vpairs = []
        self.Epairs = []

    def replace_connections(self, parent):
        """
        INPUT 
            - name of the parent box which will eat this box

        DESCRIPTION:

            - Replaces all connections to this box with the parent box EXCEPT when the parent is already connected with this vertex
            - Deletes this box from vpair and epair lists
        """

        for connection in self.connections:

            if connection == parent:
                continue

            box = self.get_box_object(connection)

            if not parent in box.connections:
                box.connections[parent] = box.connections[self.name]

            if self.name in box.connections:
                box.connections.pop(self.name)

            if self.name in box.Vpairs:
                index_to_delete = box.Vpairs.index(self.name)
                box.Vpairs.pop(index_to_delete)

            if self.name in box.Epairs:
                index_to_delete = box.Epairs.index(self.name)
                box.Epairs.pop(index_to_delete)

    def calc_Vdim(self):
        vdim = 0
        distinct_labels = []
        for connection in self.connections:
            label = self.connections[connection]
            present =False
            for distinct_label in distinct_labels:
              
                if (label == distinct_label).all():
                    present = True

            if not present:
                distinct_labels.append(label)

        self.Vdim = len(distinct_labels)

    def calc_Vdir(self):
        Vdir = [0,0,0]

        distinct_labels = []
        for connection in self.connections:
            label = self.connections[connection]
            present =False
            for distinct_label in distinct_labels:
                if (label == distinct_label).all():
                    present = True

            if not present:
                distinct_labels.append(label)

        for label in distinct_labels:

            Vdir[0] += label[0]
            Vdir[1] += label[1]
            Vdir[2] += label[2]

        self.Vdir = Vdir
        
    def get_potential_neighbours_names(self):
        name_split = self.name.split("_")
        x = int(name_split[-3])
        y = int(name_split[-2])
        z = int(name_split[-1])

        neigbours = ['Box_' +str(x+1) + "_" + str(y) + "_" + str(z),
                    'Box' + "_" + str(x-1) + "_" + str(y) + "_" + str(z),
                    'Box' + "_" + str(x) + "_" + str(y+1) + "_" + str(z), 
                    'Box' + "_" + str(x) + "_" + str(y-1) + "_" + str(z), 
                    'Box' + "_" + str(x) + "_" + str(y) + "_" + str(z+1), 
                    'Box' + "_" + str(x) + "_" + str(y) + "_" + str(z-1)]
        return neigbours  

    def get_surrounding_boxes(self, box_name):
        """
        Gets all the boxes of enlarged cube (i.e. one octree subdivision level higher)
        """
        name_split = box_name.split("_")
        x_base = int(name_split[-3])
        y_base = int(name_split[-2])
        z_base = int(name_split[-1])
        
        x,y,z = np.mgrid[ x_base - 1 : x_base + 2 : 1, y_base -1 : y_base + 2 : 1, z_base -1 : z_base + 2 : 1]
        xyz_stack = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

        neighbours = []

        parent_dict = eval(self.dict_name)

        for xyz in xyz_stack:
            
            neighbour_name = 'Box_' +str(xyz[0]) + "_" + str(xyz[1]) + "_" + str(xyz[2])

            if neighbour_name in parent_dict:
                neighbours.append(neighbour_name)
            
        
        #Delete ourselves from the list
        if self.name in neighbours:
            index_to_delete= neighbours.index(self.name)
            neighbours.pop(index_to_delete)

        return neighbours  

    def get_directional_labels(self, neighbours):
        """
        Returns dict of the labels associated with the given neighbours
        i.e.: 

        labels = {"Box001": [0,1,0], ...., "Box021": np.array([0,0,-1]) }
        +/-x = (-/+1, 0, 0)
        +/-y = (0, -/+1, 0)
        +/-z = (0, 0, -/+1)
        """

        name_box = self.name.split("_")
        x = int(name_box[-3])
        y = int(name_box[-2])
        z = int(name_box[-1])

        labels = {}
        for neighbour in neighbours:
            name_neighbour = neighbour.split("_")
            x_n = int(name_neighbour[-3])
            y_n = int(name_neighbour[-2])
            z_n = int(name_neighbour[-1])

            #X should be negative when current x is HIGHER then the neighbours x:
            labels[neighbour] = np.array([x_n-x, y_n-y, z_n-z])
        return labels

    def get_box_object(self, box_name):
        #Returns the actual class object whithe name 
        #If box object does not exists it returns false
        try:
            box_object = eval("{}['{}']".format(self.dict_name, box_name))
        except KeyError:
            box_object = False

        return box_object

    def get_neighbour_names(self):
        """
        Returns list of neighbour names as list of strings
            I.E. ["Box002", ...,  "Box003"]
        """
        neighbour_names = self.get_potential_neighbours_names()

        neighbours = []
        parent_dict = eval(self.dict_name)
        for neighbour_name in neighbour_names:
            #If name inside the dict it exists and thus is a valid neighbour
            if neighbour_name in parent_dict:
                neighbours.append(neighbour_name)
        return neighbours

    def calc_median_distance(self, points, cg, normal_vec):
        '''
        Calculates the median of the SQUARED distances

        INPUTS
            points: points calculate the distances in Nx3 format
            cg: position of the plane to which we calculate the distance
            normal_vec: the vector normal the plane 
        
        ACTIONS:
            1) Calculate all the distances
            2) return the median value
        '''

        distances = []

        #in ax +by + cZ = d, this is d
        d = np.dot(normal_vec, cg)

        #a,b,c are the entries of normal vec, respectively
        for point in points:
            distance_to_plane = (np.dot(normal_vec, point) - d )**2
            distances.append(distance_to_plane)

        return np.mean(distances)

    def check_connection_criteria(self,c1,c2,points1,points2,treshold):

        c12 = c1 + (c2-c1)*0.5 

        normal_plane_vec = (c2-c1)/np.linalg.norm((c2-c1))

        d1 = self.calc_median_distance(points1, c1, normal_plane_vec)

        d2 = self.calc_median_distance(points2, c2, normal_plane_vec)
    
        d12 = self.calc_median_distance( np.concatenate((points1, points2), axis = 0), c12, normal_plane_vec)

        #If meets critera: add connection to both the boxes
        if treshold*d12 <= min(d1,d2):
            return True
        else:
            return False

    def find_super_boxes(self, levels_higher):
        """
        The octree is a cube divided by 8 and the resultant cubes divided by 8 etc. Thus it has 8^N squares where N is the subdivison level.
        Here we choose to get a certain super box which ensures that during level 2 this specific box is never on the edge

        RETURNS:
            - list with 1 superboxes i.e. super_box:[name1, name2,name3] 
                level == 1 --> super box lower right, extended forward
                level == 2 --> super box of level 1 taken to be the upper left, extended backward
        """
        name_split = self.name.split("_")
        x_base = int(name_split[-3])
        y_base = int(name_split[-2])
        z_base = int(name_split[-1])


        if levels_higher == 1:
            x,y,z = np.mgrid[ x_base -1 : x_base +1 : 1, y_base : y_base + 2 : 1, z_base : z_base + 2 : 1]
        elif levels_higher ==2:
            x,y,z = np.mgrid[ x_base - 1 : x_base + 3 : 1, y_base -2 : y_base + 2 : 1, z_base -2 : z_base + 2 : 1]

        xyz_stack = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

        super_box = [] 
        for x,y,z in xyz_stack:
            box_name = "Box_" + str(x) + "_" + str(y) + "_" + str(z)
            super_box.append(box_name)


        return super_box

    def find_adjacent_super_boxes(self, super_box, levels_higher):
        """
        Finds the adjacent superboxes for this super box
        """

        x_min = 99999999; x_max =0
        y_min = 99999999; y_max =0
        z_min = 99999999; z_max =0

        #First find minimum x,y,z
        for box_name in super_box:

            name_split = box_name.split("_")
            x = int(name_split[-3])
            y = int(name_split[-2])
            z = int(name_split[-1])

            if x< x_min:
                x_min = x
            elif x> x_max:
                x_max = x

            if y< y_min:
                y_min = y
            elif y> y_max:
                y_max = y

            if z< z_min:
                z_min = z
            elif z> z_max:
                z_max = z

        step_size = (2* levels_higher)
        #Make  all combinations of xyz and z values in range of +/-2 of the min and maximum x,y,z, values:
        x,y,z = np.mgrid[ x_min - step_size : x_max + (step_size + 1) : 1, y_min -step_size : y_max + (step_size+1) : 1, z_min -step_size : z_max + (step_size+1) : 1]
        xyz_stack = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

        adj_super_boxes = {'adj_super_box1': {'directional_label':[], 'box_names':[]},'adj_super_box2': {'directional_label':[], 'box_names':[]},'adj_super_box3': {'directional_label':[], 'box_names':[]}
                            ,'adj_super_box4': {'directional_label':[], 'box_names':[]}, 'adj_super_box5': {'directional_label':[], 'box_names':[]}, 'adj_super_box6': {'directional_label':[], 'box_names':[]}}
        #Now select the appropriate values from this list:
        for x,y,z in xyz_stack:

            box_name = "Box_" + str(x) + "_" + str(y) + "_" + str(z)
            #Super boxes extended from x axis
            if (x < x_min or x > x_max) and (y >= y_min and  y <= y_max) and (z >= z_min and  z <= z_max):
                
                if x< x_min:
                    adj_super_boxes['adj_super_box1']['box_names'].append(box_name)
                    adj_super_boxes['adj_super_box1']['directional_label'] = np.array([-1,0,0])
                elif x > x_max:
                    adj_super_boxes['adj_super_box2']['box_names'].append(box_name)
                    adj_super_boxes['adj_super_box2']['directional_label'] = np.array([1,0,0])

            #Super boxes extended from y axis
            if (y < y_min or y > y_max) and (x >= x_min and  x <= x_max) and (z >= z_min and  z <= z_max):
                if y < y_min:
                    adj_super_boxes['adj_super_box3']['box_names'].append(box_name)
                    adj_super_boxes['adj_super_box3']['directional_label'] = np.array([0,-1,0])
                elif y > y_max:
                    adj_super_boxes['adj_super_box4']['box_names'].append(box_name)
                    adj_super_boxes['adj_super_box4']['directional_label'] = np.array([0,1,0])

            #Super boxes extended from z axis    
            if (z < z_min or z > z_max) and (y >= y_min and  y <= y_max) and (x >= x_min and  x <= x_max):
                if z < z_min:
                    adj_super_boxes['adj_super_box5']['box_names'].append(box_name)
                    adj_super_boxes['adj_super_box5']['directional_label'] = np.array([0,0,-1])
                elif z > z_max:
                    adj_super_boxes['adj_super_box6']['box_names'].append(box_name)
                    adj_super_boxes['adj_super_box6']['directional_label'] = np.array([0,0,1])

        return adj_super_boxes

    def get_super_box_properties(self, super_box):

        points = np.array([])
        for box in super_box:
            box = self.get_box_object(box)
            
            if box:
                if points.any():
                    points = np.concatenate((points,box.points), axis = 0)
                else:
                    points = box.points

        if points.any():
            cg = np.mean(points,axis =0)
        else:
            cg = np.array([0,0,0])

        return points, cg

    def get_best_connection_to_super_box(self, super_box, directional_label):
        """
        INPUT:  
            - the directional label from this super box to this super box
            - the names of the small boxes in the adjacent super box
        OUTPUT
            - minimum distance and box_name
        DESCRIPTION:
            - Finds the closest box with points in the super box and connects to it with the given directional label
        """

        distances =[]
        potential_boxes = []
        for box in super_box:

            box = self.get_box_object(box)
            #If the box exists:
            if box:
                if box.contains_points:
                    distance_vec = (self.cg - box.cg)
                    #Squared distance
                    distance = np.dot(distance_vec, distance_vec)

                    distances.append(distance)
                    potential_boxes.append(box.name)

        #If there were any boxes with points found, i.e. if there are ANY distances
        if distances:
            #Find minimum
            index_minimum = np.argmin(distances)
            box_to_connect = potential_boxes[index_minimum]
            min_distance = distances[index_minimum]

            #Add the found connections to both boxes
            box_to_connect = self.get_box_object(box_to_connect)
            self.connections[box_to_connect.name] = directional_label
            box_to_connect.connections[self.name] = -directional_label
        else:
            min_distance = box_to_connect = False

        return min_distance, box_to_connect

    def find_connections_higher_level_box(self, threshold):
        """
        DESCRIPTION: 
            Finds the connections 1 subdivision level higher in the octree.
            i.e. combines this box with 3 other boxes == the cube at one higher level. Checks connections between this box and other higher level boxes
        ACTIONS
            1) Finds a 'super box'
            2) Finds the neighbouring 'super boxes' 
            3) checks connections
            4) If not found go the next level super box
            5) If found we connect the regular box to the clossest regular box in the connected super box.

        """
        levels_higher =1
        while len(self.connections) == 0: 
            #find super box, 8 possibilities: this box is lower right, lower left, upper right or upper left, extended forward/backward
            super_box = self.find_super_boxes(levels_higher)

            # #List the super boxes and shuffle to choose randomly:
            # keys = list(super_boxes.keys())
            # random.shuffle(keys)

            # found_connection = False
            possible_connection = {}
            adj_super_boxes = self.find_adjacent_super_boxes(super_box, levels_higher)
            points1, cg1 = self.get_super_box_properties(super_box)

            #Check if we can find a conenction between our super box and the adjacent ones:
            
            for super_box_name in adj_super_boxes:
                
                adj_super_box_names = adj_super_boxes[super_box_name]['box_names']
                directional_label = adj_super_boxes[super_box_name]['directional_label']

                points2, cg2 = self.get_super_box_properties(adj_super_box_names)

                if points2.any():
                    connection = self.check_connection_criteria(cg1,cg2,points1,points2,threshold)
                else:
                    connection = False

                if connection:
                    distance, box_name = self.get_best_connection_to_super_box(adj_super_box_names, directional_label)
                    if box_name:
                        possible_connection[box_name] = [distance,directional_label]
            
            if levels_higher ==2:   
                break
            levels_higher +=1
        

    def find_connections(self,treshold):
        """
        INPUTS:
            threshold for the connection criteria: threshold * d12 <= min(d1,d2)

        ACTION:
            1) Find the neighbours
            2) Check if there is already a connection with this neighbour if so continue with next neighbour
            3) Check criteria for connection
            4) If we make a conncection add this connection to BOTH the boxes
            5) Calculate the new Vdim and Vdir for both the neighbour and this box

        """

        neighbour_names = self.get_neighbour_names()
        directional_labels = self.get_directional_labels(neighbour_names)

        for neighbour in neighbour_names:
            neighbour = self.get_box_object(neighbour)

            #Check if neighbour is already in connection:
            if neighbour.name in self.connections or neighbour.contains_points ==False:
                continue

            connection = self.check_connection_criteria(self.cg, neighbour.cg, self.points, neighbour.points, treshold)
        
            #If meets critera: add connection to both the boxes
            if connection:

                #Directional label form the directional_label dict:
                directional_label = directional_labels[neighbour.name]

                #Add the connection to the connections dict as {"boxname": direction}
                self.connections[neighbour.name] = directional_label
                #Do the same for the neighbour label but there the directional label is negative of what is found here
                neighbour.connections[self.name] = -directional_label
                neighbour.calc_Vdim()
                neighbour.calc_Vdir()
        if self.use_higher_dimensional_boxes:
            if len(self.connections) < 2 and self.contains_points:
                self.find_connections_higher_level_box(threshold)
        self.calc_Vdim()
        self.calc_Vdir()

    def get_combined_dim(self, neighbour):
        #First obtains the lsit of connection and labels if these 2 boxes were combined
        #Then calculates the vdim 

        total_connections = self.connections.copy()

        #delete the neighbour from the connection list
        total_connections.pop(neighbour.name)

        #Add all the connection of the neighbour excluding the ones already present and this box itself
        for connection in neighbour.connections.keys():
            if connection != self.name and not connection in total_connections:
                total_connections[connection] = neighbour.connections[connection]

        #calulate vdim:
        vdim = 0
        distinct_labels = []
        for connection in total_connections.keys():

            label = total_connections[connection]

            present =False
            for distinct_label in distinct_labels:
                if (label == distinct_label).all():
                    present = True

            if not present:
                distinct_labels.append(label)

        combined_dim = len(distinct_labels)

        return combined_dim

    def find_v_pairs(self):
        """
        Checks wheter a V pair is present
        
        1) Check if neighbour is already a vpair
        2) dim(combination) <= max(self.dim, neighbour.dim)
        3) Do we have an identical neighbour i.e. name + direction
        
        """
        found_pair = False
        # print("Checking Vpairs of", self.name)
        for neighbour_name in self.connections.keys():
            neighbour = self.get_box_object(neighbour_name)

            #Check 1: is it already a vpair?
            if neighbour_name in self.Vpairs:
                continue
            #check 2:
            combined_dim = self.get_combined_dim(neighbour)
            if not (combined_dim <= max(neighbour.Vdim, self.Vdim)):
                #If false skip this neighbour
                continue

            #check 3:
            #For each neighbour we check wheter they have an identical neighbour i.e. matching in name and directional label
            for connection_neighbour in neighbour.connections.keys():
                #matching name?
                if connection_neighbour in self.connections.keys():
                    
                    #matching directional label?
                    if (neighbour.connections[connection_neighbour] == self.connections[connection_neighbour]).all():
                        #This neighbour is a Vpair:
                        self.Vpairs.append(neighbour_name)
                        # neighbour.Vpairs.append(self.name)                    
                        found_pair = True

        return found_pair

    def find_e_pairs(self):
        """
        Checks wheter a E pair is present
        
        1) Check if neighbour is already a epair
        2) dim(combination) <= max(self.dim, neighbour.dim)
        3) vdir is NOT [0,0,0]
        4) is the connection with the neighbour in the same direction of one of the nonzero entries of vdir?
        5) The Epair does not form a line
        """
        check1 = 0; check2 = 0; check3 = 0; check4 = 0; check5 = 0

        found_pair = False
        for neighbour_name in self.connections.keys():

            neighbour = self.get_box_object(neighbour_name)

            #Check1  is it already a Epair or is it already merged??
            if neighbour_name in self.Epairs or neighbour.merged:
                continue
            #check 2:
            combined_dim = self.get_combined_dim(neighbour)
            if not (combined_dim <= max(neighbour.Vdim, self.Vdim)):
                check2+=1
                continue
            
            #check 3:
            if (self.Vdir.count(0) == 3):
                check3+=1
                continue
            
            #check 4:
            connection_direction = self.connections[neighbour_name]
            index_connection_direction = np.nonzero(connection_direction)[0]
            non_zero_indices_Vdir = np.nonzero(self.Vdir)[0]

            if not index_connection_direction in non_zero_indices_Vdir:
                check4+=1
                continue
            
            #check 5:
            #Does this pair form a line? I.E. one of them only has 2 connections or less
            if len(self.connections)<3 or len(neighbour.connections) <3:
                check5+= 1
                continue

            #Passed all checks thus it is an Epair
            self.Epairs.append(neighbour.name)
            # neighbour.Epairs.append(self.name)

            # print("Found EPAIR connecting:", self.name, "and", neighbour.name)
            found_pair = True

        return found_pair

    def get_best_epair(self, box_name):
        
        smallest_norm = 999
        best_Epair = ""

        box = self.get_box_object(box_name)

        for Epair in box.Epairs:

            # if Epair == requester_name:
            #     continue
            Epair = self.get_box_object(Epair)

            #If Epair is not already merged check it
            if Epair.merged:
                #If merged delte it from the Epair list
                Epair_index = box.Epairs.index(Epair.name)
                box.Epairs.pop(Epair_index)
                
            else:
                #Norm of vdir
                norm = abs(Epair.Vdir[0]) + abs(Epair.Vdir[1]) + abs(Epair.Vdir[2])
                # print(norm)

                if norm < smallest_norm:
                    best_Epair = Epair.name
                    smallest_norm = norm

        return best_Epair, smallest_norm

    def get_best_vpair(self, box_name, requester_name):
        
        smallest_norm = 999
        best_Vpair = ""

        box = self.get_box_object(box_name)

        for Vpair in box.Vpairs:

            if Vpair == requester_name:
                continue
            Vpair = self.get_box_object(Vpair)

            #If Vpair is not already merged check it
            if Vpair.merged:
                Vpair_index = box.Vpairs.index(Vpair.name)
                box.Vpairs.pop(Vpair_index)
                # print(Vpair.name, "is merged with", Vpair.parent, " Deleted index", Vpair_index)
            else:

                #Norm of vdir
                norm = abs(Vpair.Vdir[0]) + abs(Vpair.Vdir[1]) + abs(Vpair.Vdir[2])
                # print(norm)

                if norm < smallest_norm:
                    best_Vpair = Vpair.name
                    smallest_norm = norm

        return best_Vpair, smallest_norm

    def eat_v_pair(self):
        """
        Here we merge the Vpairs
        INPUTS:
            self
        
        RETURNS:
            List of new Vpairs
        
        ACTION:
            1) Eat one of them
            2) Then delete the vpair from the Vpair list 
        """
        ate_vpair = False
        for Vpair in self.Vpairs:

            self.eat_box(Vpair)
            ate_vpair = True
            break

        return ate_vpair

    def eat_e_pair(self):
        """
        Here we merge the Vpairs
        INPUTS:
            self
        
        RETURNS:
            True if E pair was eaten
        
        ACTION:
            1) Get the best Epair
            2) EAT IT
            3) Delete the Epiar from the EPAIR list
        """
        ate_Epair = False
        
        best_Epair, norm = self.get_best_epair(self.name)

        #If we found an Epair we have to check if they themselve dont have a better epair
        if best_Epair:
            #Now check if this node has a better epair:
            box = self.get_box_object(best_Epair)

            box_best_epair, box_norm = self.get_best_epair(box.name)

            #This box has a better epair when it finds a normal smaller then we found
            if norm <= box_norm or box_best_epair == self.name:
                ate_Epair = True
                # print("Found Epair to eat:", self.name, "Should eat", box.name)
                #Eat it 
                self.eat_box(box.name)


        return ate_Epair

    def eat_box(self, box_name):
        """
        Here we merge the given box with this one making this the parent box
        INPUTS:
            box_name: Box to eat

        RETURNS:
            True or False based on if ANY new Vpairs were created during this action
        ACTION:
           
            1) Will add box_to_eat to the children of this box
            2) Will remove the points of box_to_eat and add them to this box
            3) Will redirect all connections of box_to_eat to THIS box
            4) Will calculate the new C.G based on all the new points
                The paper says to use: CG_new = (W1 * CG1* + W2*CG2) / (W1+W2)
                where W1 and W2 are equal to the number of points  in the corresponding boxes
                But they mention subdivision levels and such so lets just keep it simple for now..
            5) Delete this box from vpairs/epairs list
            6) Set the parent and merge status of the box_to_eat to: the name of this box and `True`.
            7) Deletes the merged box from the dictionary of boxes

        """
        
        #Get box to eat
        box_to_eat = self.get_box_object(box_name)

        #Add the new points to this box
        if box_to_eat.contains_points:
            self.points = np.concatenate((self.points, box_to_eat.points), axis = 0)
        #Append the new child to the child list as well as its own children
        self.children.append(box_to_eat.name)
        for child in box_to_eat.children:
            self.children.append(child)
        #Calc new cg
        self.calc_cg()

        """
        Add connections and change the old connection to box_to_eat to his now parent

        1) Add all the connections of box_to_eat to his parents connections if they are not already present
        2) Change THEIR connection (so the boxes refering to box_to_eat now refer to its parent) and delete the old connections
        """
        for connection in box_to_eat.connections.keys():
            #Skip if its our selves
            if connection == self.name:
                continue
            #skip if its already in our connection list
            if connection in self.connections:
               continue
            self.connections[connection] = box_to_eat.connections[connection]
        
        #replace all connections with box_to_eat with this parent box
        box_to_eat.replace_connections(self.name)
        #delete connection with eaten box
        if box_name in self.connections:
            self.connections.pop(box_name)

        if box_name in self.Vpairs:
            index_to_delete = self.Vpairs.index(box_name)
            self.Vpairs.pop(index_to_delete)

        if box_name in self.Epairs:
            index_to_delete = self.Epairs.index(box_name)
            self.Epairs.pop(index_to_delete)

        #recalculate the Vdir and Vdim
        self.calc_Vdir()
        self.calc_Vdim()

        #Check for new Vpairs for this vertex and all its connections
        found_vpair = self.find_v_pairs()

        for connection in self.connections:
            neighbour = self.get_box_object(connection)
            
            neighbour_found_vpair = neighbour.find_v_pairs()

            if neighbour_found_vpair:
                found_vpair = True

        #Set eaten box to merge status
        box_to_eat.merged_with(self.name)

        return found_vpair

if __name__ == "__main__":

    import random 

    nboxes = 5000 #Number of boxes in which to divide the bounding box of the given set of points

    #SAVE THE OUTPUT?
    SAVE_DICT = False
    folder = os.path.abspath("./Data")
    save_name = "myDict" #Automatically adds "_Nxxx_txx.pkl" N = number of points, t = threshold

    points = np.load("Data/simple_tree.npy")
    
    make_plot(points)

    print("Getting ~", nboxes, "boxes for the", len(points), "points!")
    t0 = time.perf_counter()
    boxes = get_boxes(nboxes,points)
    t1 =time.perf_counter()
    print("\tTime:", round(t1-t0,3),"seconds!")

    myDict = {}

    #SHUFFLEs THE BOXES: This makes sure that random Vpairs are merge and it does not have a bias to start a 0,0 and work is way up from there 
    random.shuffle(boxes)
    for box in boxes:
        box_name = box[1]
        points_box = box[0]
        myDict[box_name] = myBox("myDict", box_name, points_box)


    # #make OCTREE
    ts = [16] #[8,16,24,32,64,128]

    for t in ts:

        threshold = 1/t

        print("Getting octtree using threshold 1 /", t,"...")
        t0 = time.perf_counter()
        find_all_connections(myDict, threshold)
        t1 = time.perf_counter()
        number_of_boxes = count_boxes_with_points(myDict)
        number_of_connections = count_connections(myDict)
        print("\tFound %s boxes with points!"%number_of_boxes)
        print("\tTotal connections: %s!"%number_of_connections)
        print("\tTime:", round(t1-t0,3),"seconds!")

        cg, labels_cg = plot_boxes(myDict)
        cube_points, cube_labels = draw_cubes(boxes)
        
        
        
        print("Performing collapsing procedure...")
        t0 = time.perf_counter()

        for dim in [5,4,3,2]:
            
            dim_list = make_dim_list(myDict, dim)
            
            find_all_Vpairs(myDict, dim_list)  
            find_all_Epairs(myDict, dim_list)

            #If we are Vpairs we check for the new Vpairs and eat all of them till we have nothing mroe to eat
            vpairs_there = True
            cycles = 0

            csv_path = os.getcwd() + "/Data/NNModel/class_dict.csv"
            
            while vpairs_there:
            
                _ = eat_all_Vpairs(myDict, dim_list)

                find_all_Vpairs(myDict, dim_list)  
                vpairs_there, total = still_vpairs(myDict, dim_list)

                sys.stdout.write("\tEating Vpair cycle {}!\r".format(cycles))
                sys.stdout.flush()

                cycles+=1
                #If we didnt find any new V-pairs we merge ONE E-pair
                if vpairs_there:
                    continue
                else:

                    find_all_Epairs(myDict, dim_list)
                    succes = eat_one_epair(myDict, dim_list)
                    find_all_Vpairs(myDict, dim_list)  

                    #If there was an Epair eaten: We continue the search for Vpairs
                    if succes:
                        #Make this one to stay in the loop
                        vpairs_there = True
            # cg, labels_cg = plot_boxes(myDict, False) 
            # make_plot(np.concatenate((points,cg), axis = 0), np.concatenate((labels/255,labels_cg*2/255), axis=0))         
            print("\tFinished dim %s collapse in %s cycles!"%(dim,cycles))

            total_E= 0 
            total_V= 0 
            for box in dim_list:
                try:
                    box = myDict[box]
                except KeyError:
                    continue

                Epairs = len(box.Epairs)
                Vpairs = len(box.Vpairs)
                total_E+= Epairs
                total_V+= Vpairs

        t1 =time.perf_counter()
        print("Time:", round(t1-t0,3),"seconds!")
        cg, labels_cg = plot_boxes(myDict)

        t0 = time.perf_counter()

        # Save as pickle file
        if SAVE_DICT:
            if not os.path.isdir(folder):
                os.mkdir(folder)
                
            name_file ="%s_N%sK_t%s.pkl"%(save_name, str(nboxes)[0:2], t )
            
            f = open(folder + "/" + name_file, "wb")
            pickle.dump(myDict,f, pickle.HIGHEST_PROTOCOL)
            f.close()

            print("Saved the dict!\n\tLocation: %s"%(folder + "/" + name_file))
