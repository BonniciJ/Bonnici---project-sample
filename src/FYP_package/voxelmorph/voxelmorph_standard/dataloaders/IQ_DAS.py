import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from matplotlib import pyplot as plt
import time
import os

#For beamforming
import scipy.io
import numpy.matlib
#%%
"""
Module containing IQ demodulation and DAS

V1:

Notes:
The way IQ DAS works is as follows: each RF line is IQ demodulated. Then, the
DAS is applied, which results in a complex B-mode image. The magnitude of this
resulting signal (i.e. abs()) provides the envelope data.
"""

#%%
def plot_fft(x, timestep):
    amps = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d = timestep)
    
    fig, (ft) = plt.subplots(1, sharex=True, figsize=(10,8))
    plt.plot(freqs, amps.real, 'b-',label = 'Real part', alpha = 0.75)
    plt.plot(freqs, amps.imag, 'r--',label = 'Imaginary part', alpha = 0.75)
    fig, (ft) = plt.subplots(1, sharex=True, figsize=(10,8))
    plt.plot(freqs, abs(amps), 'k',label = 'Magnitude', alpha = 0.5)
#%%
#Low pass filter
def butter_LP(cutoff, fs, order=5):
    nyq = 0.5 * fs #Nyquist Frequency
    high = cutoff / nyq #normalised cutoff 
    sos = butter(order, high, analog=False, btype='lowpass', output='sos')
    return sos

def butter_LPF(data, cutoff, fs, order=5):
    """
    Applies the low pass filter to the data
    """
    sos = butter_LP(cutoff, fs, order=order)

    #y = sosfilt(sos, data) #NOT zero-phase filtering
    y = sosfiltfilt(sos, data) #zero-phase filtering

    #For difference between sosfilt and sosfiltfilt see:
    #https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt
    return y

#%%
def IQ_demodulation(signal, bandwidth, fs, f0, t0 = 0):
    t = np.arange(len(signal))/fs + t0
    carrier = np.exp(-2j*np.pi*f0*t)
    temp = signal * carrier
#    cutoff = bandwidth/2
    cutoff = bandwidth #for some reason this is better than 
    x = 2* butter_LPF(temp, cutoff, fs, 5) #factor of 2 ensures that the envelope
                                            #amplitude is what is expected
    
    x = x * np.exp(2j*np.pi*f0*t)
    
    #Need to implement decimation (reducing sampling rate as high frequency components are uncessary)
    return x 


def pad(v, ind):
    #From the way data is saved by Field II and indicies are calculated, might
    #need to pad the raw RF signal. This padding does so in the most resource
    #efficient way.
    
    og, _ = v.shape
    
    MIN, MAX = abs(np.floor(np.amin(ind)).astype('int')), np.ceil(np.amax(ind)).astype('int')
    MAX = MAX - og + 1
    
    if MIN > MAX:    
        return np.pad(v, ((0, MIN), (0,0)))
    else:
        return np.pad(v, ((0, MAX), (0,0)))

def linear_interpol(v, ind):
    temp = np.arange(0,v.shape[1])

    up = np.ceil(ind).astype('int')     #index
    down = np.floor(ind).astype('int')  #index

    v = pad(v, ind)

    v_u = v[up,temp]
    v_d = v[down,temp]

    actual = v_d*(up-ind) + v_u*(ind-down)

    actual = actual/(up-down)
    
    #In case up=down, no interpolation is needed
    no_interpolation = np.where(up == down)
    actual[no_interpolation] = v_u[no_interpolation]

    return actual

def IQ_DAS(pathM, framenumber, N_x = None, N_z = None, verbose = False): #is actually delay and sum of IQ data
    #Carries out beamforming with IQ signals
    #load parameters.mat using pathM
    parameters = scipy.io.loadmat(pathM+'parameters.mat')
    
    c = parameters['c'][0][0]
    N_elements = int(parameters['N_elements'][0][0])
    N_angles = int(parameters['N_angles'][0][0])
    angles_range = parameters['angles_range'][0]
    pixel_xspan = parameters['pixel_xspan'][0]
    pixel_zspan = parameters['pixel_zspan'][0]
    if N_x is None:
        N_x = int(parameters['N_x'][0][0])
    if N_z is None:
        N_z = int(parameters['N_z'][0][0])
    pathRF = pathM + 'rf_data/'
    fs = parameters['fs'][0][0]
    f0 = parameters['f0'][0][0]
    
    #Transducer setup
    pitch = 0.0002  #in m
    width = 0.00017 #in m
    kerf = pitch - width #distance between transducer elements, in m
    
    #Element positions
    elem_ix = np.arange(0, N_elements)
    
    elem_posx = elem_ix * (kerf+width)
    elem_posy = np.zeros(len(elem_posx))
    elem_posz = np.zeros(len(elem_posx))
    
    elem_pos = np.stack((elem_posx,elem_posy,elem_posz),1)
    elem_pos = elem_pos-np.average(elem_pos, axis = 0)
      
    #Angles
    if N_angles == 1:
        angles = [0]
    else:
        angles = np.linspace(angles_range[0],angles_range[1], N_angles)
    
    #Plane wave beamforming
    pixel_x = np.linspace(pixel_xspan[0],pixel_xspan[1], N_x)
    pixel_z = np.linspace(pixel_zspan[0],pixel_zspan[1], N_z)
    
    pixel_X, pixel_Z = np.meshgrid(pixel_x,pixel_z)
    pixel_Y = pixel_Z * 0 #setting all y positions to 0 because imaging in y=0 plane
    
    pixel_coords = np.stack((pixel_X, pixel_Y, pixel_Z), 2)
    pixel_coords = np.reshape(pixel_coords, [N_x*N_z, 3])
    
    #Beamforming
    if verbose:
        print('  ')
        print('Beamforming')
        print('-----------')
        start_time = time.time()
        
    all_planes = np.zeros(N_x*N_z)
    
    for j in range(len(angles)):
        alpha = angles[j]
    
        #load the data from the following path
        path = pathRF + 'frame' + str(framenumber) + '_' + 'rf_coherent_plane' + str(j+1) + '.mat'
    
        raw_data = scipy.io.loadmat(path)
        v = raw_data['v']
        t0 = raw_data['t'][0][0]
        
        #Using IQ signal for beamforming
        v_IQ = np.zeros_like(v, dtype=complex)
        for i in range(v.shape[1]):
            temp = IQ_demodulation(v[:,i], 0.8*f0, fs, f0, t0)
            v_IQ[:,i] = temp
        v = v_IQ
        
        #Delay and sum with interpolation
        #time domain indices
        ind = np.zeros([N_x*N_z, N_elements])
        
        total_pixels = pixel_coords.shape[0]
        tp10 = int(total_pixels*0.1) #total pixels 10%
        
        for jj in range(total_pixels):
            #Calculate distance from each element to each focus position
            dist = elem_pos - pixel_coords[jj, :]
            dist = np.linalg.norm(dist, axis= 1) + pixel_coords[jj, 2] * np.cos(np.deg2rad(alpha)) + pixel_coords[jj,0] *  np.sin(np.deg2rad(alpha))
            
            #transform distance to time
            delays = dist/c
            delays = delays-t0
    
            #Transform delay into samples
            indices = delays * fs
            ind[jj,:] = indices
            if jj%tp10 == 0 and verbose:
                print((jj//tp10)*10, "% of pixels complete")
            
        #Beamform
        delay = linear_interpol(v, ind)
    
        delay_and_sum = np.sum(delay, axis = 1)
        plane_img = delay_and_sum        
    
        all_planes = all_planes + plane_img
        
        #This bug was present until 11/05/2022
        # all_planes = all_planes.reshape(N_x, N_z)
        
        all_planes = all_planes.reshape(N_z, N_x)
        
    if verbose:
        print('Time to IQ demodulate and DAS: ', round(time.time()-start_time,2), 's.')
    return all_planes

#%%
# def algorithm(RF):
    #IQ envelope detection
#    IQ = IQ_demodulation(RF_Butter)
#    IQ_env = abs(IQ) #not sure envelope detection is explicitly done
    
    #Delay and sum reconstruction
#    RF_DAS = delay_and_sum(IQ_env) or IQ here
    # return RF   

def z2polar(z):
    # return np.stack((np.abs(z), np.angle(z)))
    return np.stack((np.abs(z), np.angle(z)))

def polar2z(r, theta):
    return r * np.exp( 1j * theta )

def log_compression(img, dynamic_range = 0.1):
    """
    
    dB = 20*log10(Amplitude) = 10*log10(Power)
    
    dynamic_range -> dB:
    0.1     -> -20 dB
    0.01    -> -40 dB
    0.001   -> -60 dB
    """

    MIN, MAX = np.amin(img), np.amax(img)
    img = (img-MIN)/MAX
    
    img = np.where(img<dynamic_range, dynamic_range, img) #thresholding
    
    img = 20*np.log10(img) 
    return img

def load_IQ(pathM, framenumber, US_data_type, verbose = True):    
    #Ensuring folder with data exists
    pathIQ = pathM + 'IQ_data/'
    os.makedirs(pathIQ, exist_ok=True)
    
    representation = US_data_type['representation']
    
    if representation is None:
        representation = 'aib'
    else:
        valid_names = ['aib','rtheta','rcostheta', 'envelope', 'bmode', 'bmodetheta']
        if representation not in valid_names:
            raise ValueError('Invalid representation of complex data. Valid options are: %s.'%', '.join(valid_names))

    # IQ_name_aib = pathIQ + 'aib' + '_'+ 'frame' + str(framenumber) + '.npy'
    IQ_name_aib = f'{pathIQ}aib_frame{framenumber}.npy'

    #Checks if IQ has been calculated, if not, computes and saves it
    if os.path.isfile(IQ_name_aib):
        IQ_data_aib = np.load(IQ_name_aib, allow_pickle = True)
    else:
        IQ_data = IQ_DAS(pathM, framenumber, verbose = verbose) #(2,N,N), (real, imag)
        IQ_data_aib = np.stack((IQ_data.real, IQ_data.imag)) 
        np.save(IQ_name_aib, IQ_data_aib, allow_pickle = True)        

    #IQ_DAS is time consuming, thus use aib representation as default to obtain others
    if representation == 'aib':
        return IQ_data_aib #(2,N,N), both float
    else:
        IQ_data_aib = IQ_data_aib[0] + 1j * IQ_data_aib[1] #(N,N), complex

        IQ_data = z2polar(IQ_data_aib) 
        if representation == 'rtheta':
            #(2,N,N), (r,theta)
            return IQ_data
        elif representation == 'rcostheta':
            #(2,N,N), (r,cos(theta))
            IQ_data[1] = np.cos(IQ_data[1])
            return IQ_data
        elif representation == 'envelope':
            #(1,N,N), (r)
            IQ_data = IQ_data[0][np.newaxis]
            return IQ_data
        elif representation == 'bmode':
            #(1,N,N), (bmode)
            IQ_data = log_compression(IQ_data[0], US_data_type['dynamic_range'])[np.newaxis]
            return IQ_data
        elif representation == 'bmodetheta':
            #(2,N,N), (bmode, theta)
            IQ_data[0] = log_compression(IQ_data[0], US_data_type['dynamic_range'])
            return IQ_data        