import numpy as np
import scipy.io
import os

if __package__ is None or __package__ == '': #if this script is run directly in IPython console
    # uses current directory visibility
    import motion_functions as mf
else:
    # uses current package visibility
    from . import motion_functions as mf
#%%
"""
Module that loads the deformation.
"""
#%% Functions for recovering the deformation data
def _gen_mesh(pathM):
    """
    Note: to convert XX and ZZ to (,3) pos use:
        
    pos = np.zeros((N_x*N_z,3))
    pos[:,0] = XX.flatten()
    pos[:,2] = ZZ.flatten()
    
    """
    parameters = scipy.io.loadmat(pathM+'parameters.mat')
    N_x = int(parameters['N_x'][0][0])
    N_z = int(parameters['N_z'][0][0])
    
    x_span = parameters['pixel_xspan'][0]
    dx = (x_span[1] - x_span[0])/N_x
    x = np.arange(N_x)
    x = x_span[0] + dx/2 + x * dx #centroid x-position of pixels

    z_span = parameters['pixel_zspan'][0]
    dz = (z_span[1] - z_span[0])/N_z #dy is negative
    z = np.arange(N_z)
    z = z_span[0] + dz/2 + z * dz #centroid y-position of pixels, in decreasing order

    #The real x and 'y' positions of the image.
    XX, ZZ = np.meshgrid(x,z)

    return XX, ZZ, dx, dz

def _calc_deformation(pathM):
    """
    Calculates the deformation that the phantom undertook throughtout the simulation.
    Note, the scale used is '1'.
    """
    XX, ZZ, dx, dz = _gen_mesh(pathM)
    
    deformer = mf.motion_funcs.DeformPhantom(pathM)
    
    pos = np.zeros((np.multiply(*XX.shape),3))
    pos[:,0] = XX.flatten()
    pos[:,2] = ZZ.flatten()
    
    final = deformer(pos, 1)

    deformation = final - pos #since deformer(pos,0) = pos
    
    XX_deform = deformation[:,0].reshape(XX.shape)
    ZZ_deform = deformation[:,2].reshape(XX.shape)
    
    return XX_deform, ZZ_deform, XX, ZZ, dx, dz

#%% Functions for getting the data from the path
def _add_channel_dim(XX):
    """
    Used solely by give_deformation. Adds an extra dimension (0th index) to the 
    input mesh.
    """
    assert len(XX.shape) == 2, f'Error. Dimensions of the mesh are not 2 ({len(XX.shape)}).'
    return XX[np.newaxis]

def give_deformation(pathM):
    """
    Provided path to the movie and indicies of the frames, returns the deformation field, 
    the mesh of the image and the associated dx, dz.
    
    Returns
    -------
    XX_deform, ZZ_deform, XX, ZZ: np.ndarray with shapes (1, H, W)
    
    
    """
    motion_type = np.load(pathM+'motion_parameters.npy',allow_pickle=True).item()['motion_type']
    
    if motion_type in ['tra', 'she_c', 'she_p', 'rot_c', 'rot_p']:
        raise ValueError('load_deformation currently only implemented for non-rigid deformation.')
    
    deform_dir = pathM + 'deformation/'
    
    os.makedirs(deform_dir, exist_ok = True) #creates deformation directory if doesnt exist
    
    deformation_name = f'{deform_dir}deformation.npy'
    
    if os.path.isfile(deformation_name):
        #if deformation is saved
        temp = np.load(deformation_name, allow_pickle = True).item()
        XX_deform = temp['XX_deform']
        ZZ_deform = temp['ZZ_deform']
        dx = temp['dx']
        dz = temp['dz']

        XX, ZZ, dx, dz = _gen_mesh(pathM)
        
    else:
        XX_deform, ZZ_deform, XX, ZZ, dx, dz = _calc_deformation(pathM)
        
        temp = {'XX_deform': XX_deform,
                'ZZ_deform': ZZ_deform,
                'dx': dx,
                'dz': dz}
        
        np.save(deformation_name, temp, allow_pickle = True)

    #adding channel dimension
    XX_deform = _add_channel_dim(XX_deform)
    ZZ_deform = _add_channel_dim(ZZ_deform)
    XX = _add_channel_dim(XX)
    ZZ = _add_channel_dim(ZZ)

    return XX_deform, ZZ_deform, XX, ZZ, dx, dz
