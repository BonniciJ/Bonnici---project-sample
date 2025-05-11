import numpy as np
#%%
"""
Module that generates different types of motions (periodic, aperiodic)

V1:
Generated motion profile associated with cardiac motion and breathing.

"""
#%%
#Making any custom function periodic

def make_periodic(y, desired_length):
    """
    Given a function `y`, will make it periodic (with period equal to len(y)) 
    with the length of the output equal to `desired_length`.
    
    Parameters
    ----------
    y: ndarray
       array containing values of the function that are to be made periodic
       
    desired_length: int
                    desired length of the 
    
    Return
    ------
    output: ndarray
            periodic array of values `y` with length `desired_length`.
    """
    output = []
    desired_length = int(desired_length)
    
    while len(output) < desired_length:
        output = np.concatenate((output, y), axis=None)
 
    if len(output) != desired_length:
        output = output[:desired_length]
    
    return output

def shift(y, xshift, x_stepsize):
    """
    Shifts the function `y` by `xshift`. Such that now value of y(x = xshift) is 
    located at x=0 (i.e. regular mathematical shift).
    """
    return np.roll(y, -int(xshift/x_stepsize))

#%%
#Venricular volume due to cardiac motion

def exp_decay(x, time_constant, x_shift = 0, amplitude = 1, y_shift = 0):
    return np.exp(-(x+x_shift)/time_constant)*amplitude+y_shift

def exp_charge(x, time_constant, x_shift = 0, amplitude = 1, y_shift = 0):
    return (1-np.exp(-(x+x_shift)/time_constant))*amplitude+y_shift

def cardiac_ventricular_volume(resolution = 1, increase_first = True):
    """
    Returns the normalised values for ventricular volume due to cardiac contraction.
    Length of the output is equal to the 100 * `resolution`. In the real heart,
    this whole cycle takes approximately 0.75 seconds.
    
    This approximates Figure 6.2 in Chapter 6 of `Cardiovascular Biomechanics`
    by Peter R. Hoskins et al. 
    
    Parameters
    ----------
    resolution: int
    
    increase_first: bool
                    If True, the volume first increases (as function of time)
                    If False, the volume first decreases (as function of time)
    """
    if resolution < 1:
        raise ValueError('Resolution must be an integer greater than 1')
    
    resolution = int(resolution)
    
    a = np.linspace(0,10,resolution*37)
    b = np.linspace(0,10,resolution*42)
    d = np.linspace(0,10,resolution*13)
    
    a = exp_decay(a,2)
    b = 0.75*exp_charge(b, 2, 0) + a[-1]
    c = np.linspace(b[-1],1,int(resolution*8))
    d = 1 + d * 0
    if increase_first == True:
        output = np.concatenate([b,c,d,a])
    else:
        output = np.concatenate([a,b,c,d])
        
    output = output/2 + 0.5
    return output

#%%
#Lung volume due to breathing
def shallow_breath(ns):
    TV = 0.5
    ERV = 1.1
    RV = 1.2
    t = np.linspace(0,1, int(ns/2))
    shallow = -(TV/2)*np.cos(2*np.pi*t) + ERV + RV + TV/2
    return shallow

def deep_breath(ns):
    IRV = 3.2
    TV = 0.5
    ERV = 1.1
    RV = 1.2
    t = np.linspace(0,1, int(ns/2))

    deep_duration = 1.5
    deep_t = np.linspace(0, deep_duration, ns)
    deep_amp = np.sin(2*np.pi*deep_t/3)
    
    deep = -deep_amp*np.cos(2*np.pi/2*deep_t+np.pi/2) 
    deep = IRV*ERV*deep + ERV + RV
    
    deep_recovery_duration = 1/6 # = 0.25/1.5
    deep_recovery_t = np.linspace(0, deep_recovery_duration, int(ns/10+1))
    deep_recovery_amp = np.cos(4*np.pi*deep_recovery_t/3)
    deep_recovery = deep_recovery_amp*np.sin(2*np.pi*deep_recovery_t) 
    deep_recovery = 0.75*deep_recovery[1:]+ ERV + RV
    deep_recovery2 = (TV/2)*np.cos(2*np.pi*t[:int(ns/4)]) + ERV + RV + TV/2
    deep_recovery = np.concatenate([deep_recovery,deep_recovery2])
    
    deep = np.concatenate([deep,deep_recovery])
    return deep

def lung_volume(total_breaths, ns = 100):
    """
    Returns normalised values for lung volume over respiration cycle, including
    deep breaths that occur after 5-7 (chosen randomly) shallow breaths.
    
    Parameters
    ----------
    total_breaths: int
                   total number of breaths
    
    ns: int
        resolution of the motion
    """
    ns = int(ns)
    shallow = shallow_breath(ns)
    deep = deep_breath(ns)
    
    breath = []
    random = np.random.randint(5,8)
    while total_breaths > len(breath):
        breath.append(shallow)
        random -= 1
        if random == 0:
            random = np.random.randint(5,8)
            breath.append(deep)

    breath = np.concatenate(breath)
    return breath/max(breath)