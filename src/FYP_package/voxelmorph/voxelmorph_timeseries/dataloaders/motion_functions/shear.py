import numpy as np

if __package__ is None or __package__ == '': #if this script is run directly in IPython console
    import utils as utils     # uses current directory visibility
else:
    from . import utils  # uses current package visibility 

def shear(pos, shear_axis, lambda_axis, shearing_costant, static_point):
    """
    Shears object by `shearing_costant` around `static_point` along `shear_axis` with respect to `lambda_axis`.
    
    Parameters
    ----------
    shear_axis, lambda_axis: int/(3) ndarray
          if int: 0, 1 or 2. 0: x axis, 1: y axis, 2: z axis.
          
          if (3) ndarray: axes defined by supplied vector. 
          If shear_axis and lambda_axis not perpendicular, 
          shear_axis remains unchanged and only lambda_axis' 
          projection perpendicular to shear_axis is used.
          
          shear_axis, lambda_axis cannot be the same axis
    
    shearing_costant: float
              constant by which positions along `lambda_axis` are multiplied
              
    static_point: (3) ndarray
                  central point of shearing
    
    Return
    ------
    temp: (,3) ndarray
          phantom positions that underwent shearing
     
    Examples
    --------
    if shear_axis = 0 (i.e. x axis) and lambda_axis = 2 (e.i. z axis), denoting shearing_costant as c and static_point = [0,0,0], then:
    x' = x + c * z
    y' = y 
    z' = z
    where x',y',z' are the outputs
    """
    temp = pos.copy()
    
    temp = temp - static_point
    
    if (type(shear_axis) == int) and (type(shear_axis) == int):
        temp[:,shear_axis] = temp[:,shear_axis] + shearing_costant*temp[:,lambda_axis]
    else:
        lambda_axis = utils.unit_vector(lambda_axis)
        shear_axis = utils.unit_vector(shear_axis)
        lambda_axis = lambda_axis - np.dot(shear_axis,lambda_axis) * shear_axis #determines part of lambda_axis perpendicular to shear_axis
        lambda_axis = utils.unit_vector(lambda_axis)
        
        R_shear = utils.rotation_rodrigues(shear_axis,[1.,0.,0.]) #aligns `shear_axis` with x axis
        temp = utils.apply_rotation_matrix(R_shear,temp)
        lambda_axis = np.matmul(R_shear,lambda_axis) #lambda_axis also rotates
        
        if (np.sum(np.cross(lambda_axis,[0.,0.,1.])) == 0) and (np.sum(lambda_axis+np.array([0.,0.,1.])) == 0): 
            R_lambda = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]) #if after the rotatation, the lambda_axis is antiparallel to z-axis, returns R associated with pi/2 rotation around x axis
        else:
            R_lambda = utils.rotation_rodrigues(lambda_axis,[0.,0.,1.]) #aligns `lambda_axis` with z axis
        temp = utils.apply_rotation_matrix(R_lambda,temp)
        
        temp[:,0] = temp[:,0] + shearing_costant*temp[:,2] #applies shearing if shear_axis is x axis and lambda_axis is z axis
        
        R_lambda_inv = R_lambda.transpose()
        temp = utils.apply_rotation_matrix(R_lambda_inv,temp)
        
        R_shear_inv = R_shear.transpose()
        temp = utils.apply_rotation_matrix(R_shear_inv,temp)
    
    temp = temp + static_point
        
    return temp