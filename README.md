# skeletons-from-poincloud

This repository contains two implementations of finding skeletons from pointcloud data in Python.

Skeltree
---------
This implementation is from [this](https://www.researchgate.net/publication/226911507_SkelTre_-_Robust_skeleton_extraction_from_imperfect_point_clouds) paper.

Workings:
1) It divides the the the space containing the points into boxes and connects boxes which have points which seem to be part of the same object. 
2) Then it reduces the connections trying to maintain the general direction of the connection (the directionof the connection being in either x,y or z direction). 

L1-Medial-Skeleton
----------

This algorithm is only implemented partially, here [a link](https://www.cs.sfu.ca/~haoz/pubs/huang_sig13_l1skel.pdf) to the paper.

Workings:
1) It selects a random set of points (we will call centers) from the total number of points and iteratively moves these points towards the medial center of the points in the neighborhood but adding a 'repulsion force' for other centers close by.
2) Gradaully increasing the neighborhood size, this results in the centers lining up.
3) Then it tries to connect (make branches of the skeleton) for the centers which line up nicely. The criteria for 'lining up nicely' is based on the eigenvectors of the set of points in the respective neighborhoods.
4) It uses several complicated implementation procedures ([here](https://github.com/HongqiangWei/L1-Skeleton) is their c++ implementation to connect different branches.

Step 3 and 4 are complicated to implement properly and are not well explained in the paper and therefor eventuelly I quit working on this. I had a try at this which can be seen by setting the variable `try_make_skeleton` to `True`

Using the code
-----------
Both files `L1-skeleton.py` and `skelTree.py` will by default use the provided pointclouds in `Data` and should plot the outputs. Simply run `python 'one of the files'` and all should be good.

In both implementation one can adjust the number of centers or number of boxes used or import a different `pointcloud.npy` file at the start of the main. Search: `"__main__":` .

Requirements
------------
The following python packages are used:
- pptk
- matplotlib
- scipy
- pickle

