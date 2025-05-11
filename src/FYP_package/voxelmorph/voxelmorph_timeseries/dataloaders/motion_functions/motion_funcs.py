import numpy as np

if __package__ is None or __package__ == '': #if this script is run directly in IPython console
    import shear as shear
    import rotate as rotate
    import dr as dr
    import utils as utils     # uses current directory visibility
else:
    from . import shear
    from . import rotate
    from . import dr
    from . import utils  # uses current package visibility         
#%%
"""
Module used to spawn relevant motion functions

V1:
"""
#%%
def _cal_max_displacement(x_range, z_range, area_frac):
    """
    Maximum permitted displacement, assuming that the image will be cropped by `d'
    from each edge, while ensuring that only `area_frac' needs to be cropped.
    
    area_frac: float
        fraction of cropped area/initial area. Range between 0 and 1.
    """
    
    x = abs(np.diff(x_range))
    z = abs(np.diff(z_range))
    
    xpz = x+z
    
    d = (xpz - np.sqrt(xpz**2 - 4*x*z*(1-area_frac)))/4
    return d


#%%
class MotionParametersGenerate:
    def __init__(self, experiment_params):
        self._load_dict(experiment_params)
        self._limits_centre = np.array([np.average(self._pixel_xspan), 0, np.average(self._pixel_zspan)])
        
        self.generate_limits()
        self.generate()
        self.save()

    def _load_dict(self, dictionary):
        #setting self._name = value for all members in the dictionary
        for membr in dictionary:
            name, value = membr, dictionary[membr]
            setattr(self,'_'+name, value)
            
    def generate_limits(self):
        #Ensures that at most 1-area_fraction of phantom area leaves the image area
        if self._area_fraction is not None: #if None, then max displacement is given
            self._max_displacement = _cal_max_displacement(self._pixel_xspan, self._pixel_zspan, self._area_fraction)
        
        #distance of corner of image from centre
        xl, xu = self._pixel_xspan
        zl, zu = self._pixel_zspan
        
        self._corner_loc = np.array([[xl,0,zl],[xu,0,zl],[xl,0,zu],[xu,0,zu]])
        
        self._max_distance_from_centre = max(np.linalg.norm(self._corner_loc, axis = 1))
            
        #in [m], Translation
        self._max_vector = self._max_displacement
        
        #in radians, ensures that max displacement not exceeded
        self._max_rotation = self._max_displacement/self._max_distance_from_centre
        
        #in [m], dr    
        self._max_dr = self._max_displacement

        self._image_small_side = min(abs(np.diff(self._pixel_xspan)), abs(np.diff(self._pixel_zspan)))

    def generate(self):
        self._motion_params ={'motion_type': self._motion_type}
        
        if self._motion_type == 0: #translation
            vector = utils.random_unit_vector(self._in_plane) * self._max_vector
            self._motion_params['vector'] = vector
            self._motion_params['motion_type'] = 'tra'
                                   
        elif self._motion_type == 1: #shearing about centre
            shear_axis = utils.random_unit_vector(self._in_plane)
            lambda_axis = utils.random_unit_vector(self._in_plane)

            #determines part of lambda_axis perpendicular to shear_axis
            lambda_axis = lambda_axis - np.dot(shear_axis,lambda_axis) * shear_axis 
            lambda_axis = utils.unit_vector(lambda_axis)

            #Shearing constant which ensures max_displacement is not exceeded
            shearing_costant = self._max_displacement/max(abs(np.dot(self._corner_loc, lambda_axis)))

            self._motion_params['shearing_costant'] = shearing_costant
            self._motion_params['shear_axis'] = shear_axis
            self._motion_params['lambda_axis'] = lambda_axis
            self._motion_params['static_point'] = self._limits_centre
            self._motion_params['motion_type'] = 'she_c'
            
        elif self._motion_type == 2: #shearing about a point
            shear_axis = utils.random_unit_vector(self._in_plane)
            lambda_axis = utils.random_unit_vector(self._in_plane)

            #determines part of lambda_axis perpendicular to shear_axis
            lambda_axis = lambda_axis - np.dot(shear_axis,lambda_axis) * shear_axis 
            lambda_axis = utils.unit_vector(lambda_axis)

            #Point of Shearing from image centre
            PoS_distance = np.random.uniform(self._image_small_side * 0.1, self._image_small_side * 0.9)
            PoS = utils.random_unit_vector(self._in_plane)*PoS_distance 
            
            #Shearing constant which ensures max_displacement is not exceeded
            shearing_costant = self._max_displacement/max(abs(np.dot(PoS-self._corner_loc, lambda_axis)))
            
            #Point of shearing in actual coordinates (instead of relative to image centre)
            PoS = PoS + self._limits_centre
            
            self._motion_params['shearing_costant'] = shearing_costant
            self._motion_params['shear_axis'] = shear_axis
            self._motion_params['lambda_axis'] = lambda_axis
            self._motion_params['static_point'] = PoS #point of shearing
            self._motion_params['motion_type'] = 'she_p'
            
        elif self._motion_type == 3: #rotation about centre
            rotation_angle = self._max_rotation #angle of rotation
            
            if self._in_plane == True:
                rotation_axis = np.array([0,1,0])
            else:
                rotation_axis = utils.random_unit_vector(self._in_plane)
            
            self._motion_params['rotation_axis'] = rotation_axis
            self._motion_params['rotation_angle'] = rotation_angle
            self._motion_params['static_point'] = self._limits_centre #point of rotation
            self._motion_params['motion_type'] = 'rot_c'
            
        elif self._motion_type == 4: #rotation about a point            
            PoR_distance = np.random.uniform(self._image_small_side * 0.1, self._image_small_side * 0.9)
            PoR = utils.random_unit_vector(self._in_plane) * PoR_distance #displacement from image centre
            distance_to_corners = PoR - self._corner_loc

            #maximum displacement between PoR and corners
            temp = max(np.linalg.norm(distance_to_corners,axis =1))
            
            rotation_angle = self._max_displacement/temp #angle of rotation

            #location of PoR in the coordinates (instead of relative to centre of image)
            PoR = PoR + self._limits_centre

            if self._in_plane == True:
                rotation_axis = np.array([0,1,0])
            else:
                rotation_axis = utils.random_unit_vector(self._in_plane)

            self._motion_params['rotation_axis'] = rotation_axis
            self._motion_params['rotation_angle'] = rotation_angle
            self._motion_params['static_point'] = PoR #point of rotation
            self._motion_params['motion_type'] = 'rot_p'
        
        elif self._motion_type == 5: #dr
            deform_f = dr.NonRigidGenerate(pathM = self._pathM, 
                                                max_dr = self._max_dr, 
                                                limits_x = self._pixel_xspan, 
                                                limits_y = self._pixel_yspan, 
                                                limits_z = self._pixel_zspan, 
                                                in_plane = self._in_plane, 
                                                no_y_dependance = self._no_y_dependance, 
                                                without_constant_term = self._without_constant_term,
                                                minimum_separation = self._wavelength)

            dr_params = deform_f.save(None, return_only=True) #the dr.NonRigidGenerate creates dict that can be used to recover all relevant parameters
            self._motion_params.update(dr_params)
            self._motion_params['motion_type'] = 'dr'
    
    def save(self):
        self._motion_params['max_displacement'] = self._max_displacement
        np.save(str(self._pathM) + 'motion_parameters', self._motion_params, allow_pickle = True)

#%%
class DeformPhantom:
    """
    Deformer class. When called returns the deformed `pos' by `scale' using the loaded
    deformation information.
    """
    
    def __init__(self, pathM):
        """
        Assumes all parameters have been generated, including ones used for non-ridig deformation
        """
        self.pathM = pathM            
        self.load_motion()
        
    def _load_dict(self, dictionary):
        #setting self._name = value for all members in the dictionary
        for membr in dictionary:
            name, value = membr, dictionary[membr]
            setattr(self,'_'+name, value)
    
    def load_params(self):
        self._motion_params = np.load(self.pathM +'motion_parameters.npy', allow_pickle='TRUE').item()
        self._load_dict(self._motion_params)
    
    def load_motion(self):
        self.load_params()

        if self._motion_type == 'tra':
            func = self._motion_params['vector'] #to avoid searching dict every call
            self.deform_f = lambda pos, scale: pos + func * scale
        elif self._motion_type in ['she_c', 'she_p']:
            self.deform_f = lambda pos, scale: shear.shear(pos,
                                                           shear_axis = self._motion_params['shear_axis'],
                                                           lambda_axis = self._motion_params['lambda_axis'],
                                                           shearing_costant = scale * self._motion_params['shearing_costant'],
                                                           static_point = self._motion_params['static_point'])
        elif self._motion_type in ['rot_c', 'rot_p']:
            self.deform_f = lambda pos, scale: rotate.rotate(pos,
                                                             axis = self._motion_params['rotation_axis'],
                                                             angle = scale*self._motion_params['rotation_angle'],
                                                             static_point = self._motion_params['static_point'])
        elif self._motion_type == 'dr':
            func = dr.NonRigidLoad(self.pathM)
            self.deform_f = lambda pos, scale: pos + func(pos) * scale
        else:
            raise ValueError('Invalid motion type')
    
    def __call__(self, pos, scale):
        return self.deform_f(pos, scale) #returns deformed positions
#%%
#Unit tests
def unit_test(pos, experiment_params, scale = 1):    
    #'save' initial positions
    original_pos = pos.copy()

    #### Would be done in phantom_setup
    #Create motion parameters
    MotionParametersGenerate(experiment_params)
    
    #Load deformer
    deformer = DeformPhantom(experiment_params['pathM'])
    
    #Move the positions
    new_pos = deformer(pos, scale)
    #### Would be done in phantom_setup
    
    #### Would be done when loading the deformation field for GT
    #Load deformer again
    deformer = DeformPhantom(experiment_params['pathM'])
    
    #Move the positions
    new2_pos = deformer(pos, scale)
    #### Would be done when loading the deformation field for GT

    #compare the two moved phantoms
    error = np.sum(abs(new_pos-new2_pos))

    print('Motion type %s (%s)'%(experiment_params['motion_type'], deformer._motion_type))
    print('')
    
    max_permitted_displacement = _max_displacement(experiment_params['pixel_xspan'],experiment_params['pixel_zspan'],experiment_params['area_fraction'])
    actual_max_displacement = max(np.linalg.norm(original_pos-new_pos, axis=1))
    print('Maxmim permitted displacement (a.u.): %.6f' % max_permitted_displacement)    
    print('Actual maxmim displacement (a.u.): %.6f' % actual_max_displacement)
    print('Permitted displacement > maximum displacement (should be True)?', max_permitted_displacement>actual_max_displacement)
#    print('Difference between permitted and actual max displacements', max_permitted_displacement-actual_max_displacement)
    print('')
    
    print('Demostrating that motion actually took place:')
    print('Moved distance (a.u.):', np.sum(abs(original_pos-new_pos)))
    print('')

    print('Error between two instances of loaded phantoms (should be zero)')
    print('Error: %.4f' %error)
    print('')

    print('Verifying that underlying pos is not moved, only a copy is:')
    print('Error on phantom:', np.sum(abs(original_pos-pos)))
    
    single_point = pos[0].reshape(1,3)
    print(single_point, '- single point (sp)')
    sp2 = deformer(single_point, scale)
    print(single_point, '- sp after deform (should be same as above)')
    print(sp2, '- deformed sp')
    
    
    print('------------------------')
    
def unit_test_params(motion_type):
    experiment_params = {}
    experiment_params['area_fraction'] = 0.64
    experiment_params['pathM'] = './'
    experiment_params['pixel_xspan'] = [-0.015,0.015] #in [m], have to be floats
    experiment_params['pixel_yspan'] = [-0.00015,0.00015] #in [m], have to be floats
    experiment_params['pixel_zspan'] = [0.03,0.06]#in [m], have to be floats
    experiment_params['no_y_dependance'] = True
    experiment_params['in_plane'] = True
    experiment_params['without_constant_term'] = True
    experiment_params['motion_type'] = motion_type
    
    f0=5e6               #Transducer center frequency [Hz], has to be a float
    c=1540.              #Speed of sound [m/s], has to be a float
    wavelength = c/f0
    experiment_params['wavelength'] = wavelength #in [m]
    return experiment_params

def unit_tests():    
    #Create phantom
    motion_params = unit_test_params(1)
    x_lims = motion_params['pixel_xspan']
    y_lims = motion_params['pixel_yspan']
    z_lims = motion_params['pixel_zspan']
    
    N = int(1e4) #number of scat
    pos = np.stack((np.random.uniform(*x_lims, N),np.random.uniform(*y_lims, N),np.random.uniform(*z_lims, N)),axis=1)
    
    motion_type = 0 #tra
    experiment_params = unit_test_params(motion_type)
    unit_test(pos, experiment_params)

    motion_type = 1 #she_c
    experiment_params = unit_test_params(motion_type)
    unit_test(pos, experiment_params)    

    motion_type = 2 #she_p
    experiment_params = unit_test_params(motion_type)
    unit_test(pos, experiment_params)    
    
    motion_type = 3 #rot_c
    experiment_params = unit_test_params(motion_type)
    unit_test(pos, experiment_params)
    
    motion_type = 4 #rot_p
    experiment_params = unit_test_params(motion_type)
    unit_test(pos, experiment_params)    

    motion_type = 5 #dr
    experiment_params = unit_test_params(motion_type)
    unit_test(pos, experiment_params)
    
if __name__ == '__main__':
    unit_tests()
    import os
    os.remove('./motion_parameters.npy')