import numpy as np

if __package__ is None or __package__ == '': #if this script is run directly in IPython console
    import utils as utils     # uses current directory visibility
else:
    from . import utils  # uses current package visibility 
#%%
def rotate(pos, axis, angle, static_point):
    """
    Rotates object by angle along `axis` in counter-clockwise direction around `static_point`
    
    Parameters
    ----------
    axis: int/(3) ndarray
      if int: 0, 1 or 2. 0: x axis, 1: y axis, 2: z axis.
      if (3) ndarray: axis defined by supplied vector
      
    angle: float
           angle by which rotation is made in the counter-clockwise direction is made
           In radians. 
           
    static_point: (3) ndarray
                  point around which rotation occurs
    Return
    ------
    temp: (,3) ndarray
          phantom positions that underwent rotation
    """
    temp = pos.copy()
    
    temp = temp - static_point
    cos, sin = np.cos(angle), np.sin(angle)    
    
    if type(axis) == int:
        if axis == 0:
            x = temp[:,0]
            y = temp[:,1] * cos - temp[:,2] * sin
            z = temp[:,1] * sin + temp[:,2] * cos
        elif axis == 1:
            x = temp[:,0] * cos + temp[:,2] * sin
            y = temp[:,1]
            z = -temp[:,0] * sin + temp[:,2] * cos
        elif axis == 2:
            x = temp[:,0] * cos - temp[:,1] * sin
            y = temp[:,0] * sin + temp[:,1] * cos
            z = temp[:,2]
        temp[:,0] = x
        temp[:,1] = y
        temp[:,2] = z
        
    else:
        R = utils.rotation_rodrigues(axis,[1.,0.,0.]) #aligns `axis` with x axis
        temp = utils.apply_rotation_matrix(R,temp)            
        
        y = temp[:,1] * cos - temp[:,2] * sin #applies rotation matrix around x axis, then returns back the phantom
        z = temp[:,1] * sin + temp[:,2] * cos
        temp[:,1] = y
        temp[:,2] = z
        
        R_inv = R.transpose()
        temp = utils.apply_rotation_matrix(R_inv,temp)
        
    temp = temp + static_point
    return temp