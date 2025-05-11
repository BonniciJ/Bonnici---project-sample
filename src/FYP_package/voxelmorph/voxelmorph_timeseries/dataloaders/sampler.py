import scipy.io

if __package__ is None or __package__ == '': #if this script is run directly in IPython console
    # uses current directory visibility
    import IQ_DAS as lim #load image
    import load_deformation as lde #load deformation
    import data_preprocess as dpp #data preprocessing
    
else:
    # uses current package visibility
    from . import IQ_DAS  as lim #load image
    from . import load_deformation as lde #load deformation
    from . import data_preprocess as dpp #data preprocessing
    
#%%
"""
Module that provides either one sample or the entire movie as samples.
"""
#%%
def scale(frame_index, frames_per_movie):
    """
    Returns scale of deformation for each frame from 0th frame
    """
    return (frame_index)/(frames_per_movie-1)

#%%
def samples(pathM, US_data_type, N_pixels_desired, network_reduction_factor, pad_cval_image, pad_cval_deform):
    """
    Returns all of the frames of the movie along with the deformation as a single sample.
    
    Return
    ------    
    frames: (M, 1or2, H, W)
        where M = number of frames in a movie
    
    scales: dict with [M] keys
        where M = number of frames in a movie
    
    XX_deform, ZZ_deform, XX, ZZ: (1, H, W)
        where M = number of frames in a movie. 
        ALL VALUES ARE IN PHYSICAL VALUES (not in pixels)
        
    dx, dz: float
        Physical distance between consequitive pixels
        
    pad_cval_image, pad_cval_deform: float
        Constant value with which the image and deform are padded. Note: mesh is not
        padded as it is extended using bilinear interpolation.
    """
    parameters = scipy.io.loadmat(pathM+'parameters.mat')
    max_frames = parameters['frames'][0][0]
    
    appendix = dpp.cal_appendix(pathM)
        
    # frames = {}
    scales = {}
    for i in range(max_frames):
        # img = lim.load_IQ(pathM, i, US_data_type = US_data_type)
        # frames[i] = img
        scales[i] = scale(i, max_frames)

    # frames = dpp.preprocess_movie(frames_dic = frames, appendix = appendix, US_data_type = US_data_type, N_pixels_desired = N_pixels_desired, network_reduction_factor = network_reduction_factor, pad_cval = pad_cval_image)
    
    XX_deform, ZZ_deform, XX, ZZ, dx, dz = lde.give_deformation(pathM)
    
    XX_deform = dpp.preprocess_deform(XX_deform, appendix = appendix, N_pixels_desired = N_pixels_desired, network_reduction_factor = network_reduction_factor, pad_cval_deform = pad_cval_deform)
    ZZ_deform = dpp.preprocess_deform(ZZ_deform, appendix = appendix, N_pixels_desired = N_pixels_desired, network_reduction_factor = network_reduction_factor, pad_cval_deform = pad_cval_deform)
    XX, ZZ = dpp.preprocess_meshes(XX, ZZ = ZZ, appendix = appendix, N_pixels_desired = N_pixels_desired, network_reduction_factor = network_reduction_factor, dx = dx, dz = dz)
    
    return None, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz
#%%
if __name__ == '__main__':
    def _plot_grid(x,y,title = None, ax=None, **kwargs):
        ax = ax or plt.gca()
        segs1 = np.stack((x,y), axis=2)
        segs2 = segs1.transpose(1,0,2)
        ax.add_collection(LineCollection(segs1, **kwargs))
        ax.add_collection(LineCollection(segs2, **kwargs))
        ax.autoscale()
       
        if title:
            plt.title(title)

    def plot(img, title = None):
        plt.figure()
        plt.imshow(img, cmap = 'gray')
        if title:
            plt.title(title)
    
    
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np
    
    path = '../../../../movies/dr_dataset001_0__2022_07_08/'

    # frames, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = samples(path, {'representation':'bmode', 'dynamic_range':0.01}, 320, 16)
    # frames, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = samples(path, {'representation':'envelope'}, 320, 16)
    # frames, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = samples(path, {'representation':'aib'}, 320, 16)
    # frames, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = samples(path, {'representation':'rtheta', 'normalise_phase':False}, 320, 16)

    US_data_type = {'representation': 'rtheta', 'normalise_phase': True} 
    frames, scales, XX_deform, ZZ_deform, XX, ZZ, dx, dz = samples(path, US_data_type = US_data_type, N_pixels_desired = 320, network_reduction_factor = 16, pad_cval_image = -1, pad_cval_deform = 0)

    US_data_type = {'representation': 'bmodetheta', 'dynamic_range': 0.01, 'normalise_phase': True}
    frames2, scales, XX_deform2, ZZ_deform2, XX, ZZ, dx, dz = samples(path, US_data_type = US_data_type, N_pixels_desired = 320, network_reduction_factor = 16, pad_cval_image = -1, pad_cval_deform = 0)


    print(np.sum(abs(frames[0][0] - frames2[0][0])))    
    print(np.sum(abs(frames[0][1] - frames2[0][1])))
    print(np.sum(abs(XX_deform-XX_deform2)))
    print(np.sum(abs(ZZ_deform-ZZ_deform2)))


    plot(frames[0][0])
    plot(frames2[0][0])
    plot(frames[0][1])
    plot(frames2[0][1])
    
    

    f = frames[0]
    
    # plot(f[0])
    if False:
        
        plot(np.flip(f, axis = 2)[0])
        
        XX = XX[:, 30:-30, 30:-30]
        ZZ = ZZ[:, 30:-30, 30:-30]
        XX_deform = XX_deform[:, 30:-30, 30:-30]
        ZZ_deform = ZZ_deform[:, 30:-30, 30:-30]
    
        plt.figure()
        _plot_grid(XX[0], ZZ[0], color="lightgrey")
        _plot_grid(XX[0]+XX_deform[0], ZZ[0]+ZZ_deform[0], color="C0")
        
        
        plt.figure()
        _plot_grid(XX[0], ZZ[0], color="lightgrey")
        _plot_grid(XX[0]+np.flip(-XX_deform, axis = 2)[0], ZZ[0]+np.flip(ZZ_deform, axis = 2)[0], color="C0")