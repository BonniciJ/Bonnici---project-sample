import numpy as np
#%%
"""
Module containing functions used by multiple motion types
"""
#%%
def calc_centroid(phantom_pos):
    """
    Calculates centroid of `phantom_pos`
    
    Parameters
    ----------
    phantom_pos: (,3) ndarray
                 array containing 3d position, such that phantom_pos[n-1] outputs [x_n,y_n,z_n] position of nth element
                                   
    Returns
    -------
    output: (,3) ndarray
            The centroid position
    """
    centroid = np.array([np.mean(phantom_pos[:,0]),np.mean(phantom_pos[:,1]),np.mean(phantom_pos[:,2])])
    return centroid

def unit_vector(v):
    """
    Returns unit vector pointing in direction on `v`
    
    Parameters
    ----------
    v: (3) ndarray
    
    Returns
    -------
    unit_v: (3) ndarray
            unit vector pointing in direction of `v`
    """
    unit_v = v/np.linalg.norm(v)
    return unit_v

def rotation_rodrigues(v, v_rot):
    """
    Determines rotation matrix necessary to rotate `v` into `v_rot`, using Rodrigues' rotation formula
    
    Parameters
    ----------
    v, v_rot: (3) ndarray
    
    Returns
    -------
    R: (3,3) ndarray
       Rotation matrix necessary to rotate v into v_rot
    """
    I = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    
    v = unit_vector(v)
    v_rot = unit_vector(v_rot)
    
    if np.sum(np.cross(v,v_rot)) == 0:  #True of collinear vectors inputted 
        if np.sum(v+v_rot) != 0:        #True if parallel vectors
            return I
        else:               #True if antiparallel
            I[1,1] = -1     #this is only the case when the two are in the x-axis or z-axis, need to modify futher
            return -I
            
    k = np.cross(v,v_rot)
    s = np.linalg.norm(k)   #sin(theta) - only true because v and v_rot are unit vectors
    c = np.dot(v,v_rot)     #cos(theta) - only true because v and v_rot are unit vectors
    k = unit_vector(k)
    
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    
    R = I + s*K + (1-c)*np.matmul(K,K)    
    return R

def apply_rotation_matrix(R,pos):
    """
    Applies rotation matrix to positions

    Parameters
    ----------
    R: (3,3) ndarray
       rotation matrix to be applied to pos
        
    pos: (,3) ndarray
         array containing 3d position, such that phantom_pos[n-1] outputs [x_n,y_n,z_n] position of nth element
         
    Return
    ------
    rotated: (,3) ndarray
             `pos` rotated by `R`
    """
    x = R[0,0] * pos[:,0] + R[0,1] * pos[:,1] + R[0,2] * pos[:,2]
    y = R[1,0] * pos[:,0] + R[1,1] * pos[:,1] + R[1,2] * pos[:,2]
    z = R[2,0] * pos[:,0] + R[2,1] * pos[:,1] + R[2,2] * pos[:,2]

    rotated = np.zeros(np.shape(pos))
    rotated[:,0] = x
    rotated[:,1] = y
    rotated[:,2] = z        
    return rotated

def random_unit_vector(in_plane = True):
    """
    Returns unit vector pointing in a random direction
    
    Parameters
    ----------
    in_plane: bool
              if True, the `y` component of the vector is zero
              
    Returns
    -------
    v: (3) ndarray
       random unit vector
    """
    v = 0                           #set to zero just to enter the while loop
    while np.linalg.norm(v) == 0:   #prevents division by zero
        v = np.random.rand(3) - 0.5
        if in_plane == True:
            v[1] = 0
    v = unit_vector(v)
    return v