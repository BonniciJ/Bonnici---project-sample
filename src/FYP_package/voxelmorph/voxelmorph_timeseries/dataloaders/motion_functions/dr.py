import numpy as np
from scipy.optimize import fsolve
#%%
"""
Module for generating non-rigid deformation

V3: 
    
> Added minimum separation between roots and the associated unit tests
"""

#%% General functions that could be used for encoding and decoding
def points_of_inflection(x, P):
    """
    Provided the polynomial `P' and starting guess positions `x', determines the
    points of inflection of the polynomial.
    """
    loc = fsolve(P.deriv(), x)
    loc = np.unique(loc)
    return loc

def polynomial_max(limits, P):
    """
    Determines the maximum value of the polynomial `P' within the `limits'.
    """
    x = np.linspace(*limits, 50)
    loc = points_of_inflection(x, P)
#    x = np.linspace(*limits, 2000) #computationally faster but less accurate
#    loc = x
    
    y = abs(P(loc))
    y_i = np.argmax(y)
    y = P(loc[y_i]) #maximum/mimum value
    
    x = loc[y_i] #x value associated with the max/min
    return x, y

def normalised_polynomial_from_roots(roots, limits):
    """
    Generates a normalised polynomial from the roots. The normalisation ensures
    that the max(abs(P(x)))=1 within the 'limits'
    """
    if len(roots) == 0:
        return np.polynomial.Polynomial(0)          #providing coefficients as input
    P = np.polynomial.Polynomial.fromroots(roots)   #providing roots as input
    _, a = polynomial_max(limits, P)
    return P/abs(a)

#%% Functions for generating roots, such that the np.diff(roots) >= min_sep for all places
def _no_collisions(c, roots, min_sep):
    """
    Checks if the candidate roots `c' collides with the existing `roots', such that
    they are separate by at least `min_sep'.
    
    With the existing implementation of _update_lows_highs, limits will always
    ensure that this is the case. However, this continues to exit just as a secondary
    safeguard.
    """
    if len(roots) == 0:
        return True

    collisions = np.invert(np.isclose(abs(c-roots),min_sep)) * (abs(c-roots) < min_sep)    
    if sum(collisions) > 0:
        return False
    else:
        return True

def _remaining_space(lows, highs, min_sep):
    """
    Returns the number of new roots that can be placed in the remaining space.
    """

    diff = highs-lows
    n = np.floor(diff/min_sep) + 1
    return sum(n)

def _update_lows_highs(x_min, x_max, roots, min_sep):
    """
    Returns segments, where new roots could be placed in, such that the distance
    between any roots is greater or equal to min_sep.
    
    if x - position of roots, | - min_sep away from root
    ----|--x--|---|--x--x--||--x-|-|-x--|--
    
    returns segments such that:
    ----|     |---|        ||           |--
    
    i.e. positions where new roots can be placed
    `||' - segment where low=high
    
    if:
    -x--|-------
    returns
        |-------
    """
    roots = np.array(sorted(roots))
    
    highs = roots.copy() - min_sep
    lows = roots.copy() + min_sep
    
    lows = np.insert(lows, 0, x_min)
    highs = np.append(highs, x_max)
    
    diff = highs-lows
    
    needs_removal = np.where(diff < 0)[0]

    lows = np.delete(lows, needs_removal)
    highs = np.delete(highs, needs_removal)
    return lows, highs

def _separated_roots(n, x_min, x_max, min_sep):
    """
    Generates `n' roots, in the range [`x_min', `x_max'], such that every root
    is at least `min_sep' from every other root apart.
    """
    roots = np.array([])
    
    if n == 0:
        return roots
    
    segments_lows = np.array([x_min])
    segments_highs = np.array([x_max])
    
    while len(roots) < n:
        candidates = np.random.uniform(segments_lows, segments_highs)
        np.random.shuffle(candidates) #randomly shuffles the order of candidates, so there is no bias towards candidates in the earlier segments
        for c in candidates:
            if len(roots) < n:
                no_collisions = _no_collisions(c, roots, min_sep)
                
                #proposed segments
                proposed_l, proposed_h = _update_lows_highs(x_min, x_max, np.append(roots, c), min_sep)
                rem_s = _remaining_space(proposed_l, proposed_h, min_sep)
                
                unfound_roots = n-len(roots) - 1 #roots that are yet to be generated
                
                if no_collisions and (rem_s >= unfound_roots):
                    roots = np.append(roots, c)
                    
                    #Updates segments with proposed segments
                    segments_lows, segments_highs = proposed_l, proposed_h
    return roots

#%% Used to generate random roots inside the phantom
def random_roots(limits, n_total = None, n_inside = None, outside_scale = 2., without_constant_term = True, min_sep = None):
    """
    Creating a random number of roots for a polynomial within the limits and outside
    these limits.
    """
    if min_sep is None:
        print('Warning: minimum_separation is None. This behaviour is highly discouraged.')
    
    if min_sep: #minimum separation of roots
        max_inside = np.floor(np.diff(limits)/min_sep) + 1 #maximum number of roots inside, given the minimum separation
        if n_inside:
            if n_inside > max_inside:
                raise ValueError(f'Number of roots requested ({n_inside}) is greater than the possible number of roots ({max_inside}).')
    
    if (n_inside is not None) and (n_total is not None): 
        if n_inside > n_total:
            raise ValueError('Number of roots inside the range must be equal or less to the total number of roots.')
    if n_total is None:
        n_total = np.random.randint(3,10)
    if n_inside is None:
        n_inside = np.random.randint(1, n_total+1)
    
    if min_sep:
        if n_inside > max_inside:
            n_inside = max_inside
    
    x_min, x_max = limits
    
    if n_inside == 0:
        raise ValueError('n_inside is zero.')
    elif min_sep:
        roots = _separated_roots(n_inside, x_min, x_max, min_sep)
    else:
        roots = np.random.uniform(x_min, x_max, n_inside)

    #Finding roots outside of the phantom
    centre = np.average([x_min, x_max])
    new_min = outside_scale*(x_min - centre) + centre
    new_max = outside_scale*(x_max - centre) + centre

    while len(roots) < n_total:
        candidate = np.random.uniform(new_min, new_max)
        if (candidate < x_min) or (candidate > x_max):
            roots = np.append(roots, candidate)
    
    #This might become an issue if min_sep is provided
    if without_constant_term:
        zero_idx = np.argmin(abs(roots))
        roots[zero_idx] = 0
        
    return roots
#%% dx, dy and dz are all instances of multivariable polynomials  
class MultiVariablePolynomialMeta():
    """
    Meta class used to generate and load the multi-variable polynomials.
    """
    def __init__(self):
        """
        Produces a MVP, which has a maximum value of 1 within the domain.
        """
        self.adjust_xyz_contributions()
        self.normalise()

    def multiply_by_constant(self, c):
        """
        Used when taking into account the relative contributions of dx, dy, dz towards dr
        """
        self.Px *= c
        self.Py *= c
        self.Pz *= c
    
    def adjust_xyz_contributions(self):
        """
        Multiplying the normalised polynomials by their maximum contribution to the da.
        """
        self.Px *= self.x_component
        self.Py *= self.y_component
        self.Pz *= self.z_component

    def normalise(self):
        """
        Normalises the MVP such that within the entire domain of pos, max(MVP) == 1
        """
        #Number of points along a given axis in the grid.
        N = 1000

        x = np.linspace(*self.limits_x, N)
        if len(self.Py.roots()) == 0: #If there is no y-dependance
            y = np.array([0])
        else:
            y = np.linspace(*self.limits_y, N)
            
        z = np.linspace(*self.limits_z, N)
        XX,YY,ZZ = np.meshgrid(x,y,z)
        
        #Changing into the appropriate format
        pos = np.zeros([XX.flatten().shape[0],3])
        pos[:,0] = XX.flatten()
        pos[:,1] = YY.flatten()
        pos[:,2] = ZZ.flatten()
        
        #Determining the maximum/mimum value inside the grid
        v = abs(self(pos))
        v_i = np.argmax(v)
        v = v[v_i] #maximum/minimum value
        
        #Normalising and multiplying by max_da
        self.Px = self.Px/v
        self.Py = self.Py/v
        self.Pz = self.Pz/v
    
    def __call__(self,pos):
        """
        pos: (N,3) np.array 
        """
        return self.Px(pos[...,0]) + self.Py(pos[...,1]) + self.Pz(pos[...,2]) 
    
class MVPGenerate(MultiVariablePolynomialMeta):
    """
    Generates Multi-Variable Polynomial from limits and other input parameters.
    """
    def __init__(self, limits_x, limits_y, limits_z, no_y_dependance = True, without_constant_term = True, minimum_separation = None):
        self.limits_x = limits_x
        self.limits_y = limits_y
        self.limits_z = limits_z
        self.no_y_dependance = no_y_dependance
        self.without_constant_term = without_constant_term
        self.minimum_separation = minimum_separation
        
        
        self.gen_polynomials()
        
        super().__init__()
                
    def gen_polynomials(self):
        """
        Generates the polynomials that form da(x,y,z).
        """
        components = np.random.uniform(0,1,3)
        
        self.rx = random_roots(limits = self.limits_x, without_constant_term = self.without_constant_term, min_sep = self.minimum_separation)
        self.Px = normalised_polynomial_from_roots(self.rx, limits = self.limits_x)
        
        if self.no_y_dependance == True:
            self.ry = np.array([])
            components[1] = 0
        else:
            self.ry = random_roots(limits = self.limits_y,without_constant_term = self.without_constant_term, min_sep = self.minimum_separation)
        self.Py = normalised_polynomial_from_roots(self.ry, limits = self.limits_y)
    
        self.rz = random_roots(limits = self.limits_z, without_constant_term = self.without_constant_term, min_sep = self.minimum_separation)
        self.Pz = normalised_polynomial_from_roots(self.rz, limits = self.limits_z)
    
        self.x_component, self.y_component, self.z_component = components/sum(components)
        
    def key_params(self):
        """
        Saving parameters that can be used to completely recover the MVP
        """
        params = {}
        
        params['rx'] = self.rx
        params['ry'] = self.ry
        params['rz'] = self.rz
        
        params['x_component'] = self.x_component
        params['y_component'] = self.y_component
        params['z_component'] = self.z_component
        
        return params

class MVPLoad(MultiVariablePolynomialMeta):
    """
    Loads Multi-Variable Polynomial from saved parameters.
    """
    def __init__(self, params, limits_x, limits_y, limits_z):
        self.limits_x = limits_x
        self.limits_y = limits_y
        self.limits_z = limits_z
        self.load_polynomials(params)
        
        super().__init__()

    def load_polynomials(self, params):
        """
        Loading parameters that are used to generate MVP
        """
        self.rx = params['rx']
        self.ry = params['ry']
        self.rz = params['rz']
    
        self.x_component = params['x_component']
        self.y_component = params['y_component']
        self.z_component = params['z_component']
        
        self.Px = normalised_polynomial_from_roots(self.rx, limits = self.limits_x)
        self.Py = normalised_polynomial_from_roots(self.ry, limits = self.limits_y)
        self.Pz = normalised_polynomial_from_roots(self.rz, limits = self.limits_z)


#%% Non-rigid motion i.e. dr(x,y,z)
class NonRigidMeta():
    """
    Meta class used to generate and load non-rigid deformation (dr), ensuring
    that max(norm()) == max_dr within the domain of the function.
    """
    def __init__(self):
        """
        Produces a dr, which has max(norm()) == max_dr within domain of the function.
        """
        self.constant = 1.
        self.normalise()
        self.constant *= self.max_dr
    
    def normalise(self):
        """
        Normalises the dr such that within the entire domain of pos, max(norm(dr)) == 1
        """
        self.dx.multiply_by_constant(self.max_dx)
        if self.params['dy'] is not None: #Would be None if no y-dependance
            self.dy.multiply_by_constant(self.max_dy)
        self.dz.multiply_by_constant(self.max_dz)
    
        #Number of points along a given axis in the grid.
        N = 1000

        x = np.linspace(*self.limits_x, N)
        if self.params['dy'] is None: #If there is no y-dependance
            y = np.array([0])
        else:
            y = np.linspace(*self.limits_y, N)
            
        z = np.linspace(*self.limits_z, N)
        XX,YY,ZZ = np.meshgrid(x,y,z)
        
        #Changing into the appropriate format
        pos = np.zeros([XX.flatten().shape[0],3])
        pos[:,0] = XX.flatten()
        pos[:,1] = YY.flatten()
        pos[:,2] = ZZ.flatten()
        
        #Determining the maximum/mimum value inside the grid
        v = np.linalg.norm(self(pos),axis = 1)
        v_i = np.argmax(v)
        v = v[v_i] #maximum/minimum value
    
        self.constant /= v
        
    def __call__(self, pos):
        """
        pos: (N,3) np.array 
        """
        return np.stack((self.dx(pos), self.dy(pos), self.dz(pos)), axis = 1) * self.constant

class NonRigidGenerate(NonRigidMeta):
    """
    Class that generates non-rigid deformation (dr) from limits and other input parameters.
    """
    def __init__(self, pathM, max_dr, limits_x, limits_y, limits_z, in_plane = True, no_y_dependance = True, without_constant_term = True, minimum_separation = None):
        self.max_dr = max_dr
        self.limits_x = limits_x
        self.limits_y = limits_y
        self.limits_z = limits_z
        self.in_plane = in_plane
        self.no_y_dependance = no_y_dependance
        self.without_constant_term = without_constant_term
        self.minimum_separation = minimum_separation
        
        self.gen()  #generating polynomials
        self.save(pathM) #saving parameters
        
        super().__init__()

    def gen(self):
        """
        Generating MVPs for dx, dy, dz. Thereby, creating components that ultimately form dr.
        """
        self.dx = MVPGenerate(self.limits_x, self.limits_y, self.limits_z, self.no_y_dependance, self.without_constant_term, self.minimum_separation)
        if self.in_plane:
            self.dy = lambda pos: np.zeros(pos.shape[0])
        else:
            self.dy = MVPGenerate(self.limits_x, self.limits_y, self.limits_z, self.no_y_dependance, self.without_constant_term, self.minimum_separation)
        self.dz = MVPGenerate(self.limits_x, self.limits_y, self.limits_z, self.no_y_dependance, self.without_constant_term, self.minimum_separation)

        #Fractional contributions of dx, dy, dz
        dxdydz = np.random.uniform(0,1,3)
        if self.in_plane:
            dxdydz[1] = 0
        dxdydz = dxdydz/sum(dxdydz)

        self.max_dx, self.max_dy, self.max_dz = dxdydz

    def save(self, pathM, return_only = False):
        """
        Saving all the parameters used to generate the non-rigid deformation.
        """
        self.params = {}
        self.params['dx'] = self.dx.key_params()
        if self.in_plane:
            self.params['dy'] = None
        else:
            self.params['dy'] = self.dy.key_params()
        self.params['dz'] = self.dz.key_params()
        
        self.params['max_dx'] = self.max_dx
        self.params['max_dy'] = self.max_dy
        self.params['max_dz'] = self.max_dz
        
        self.params['max_dr'] = self.max_dr
        
        self.params['limits_x'] = self.limits_x
        self.params['limits_y'] = self.limits_y
        self.params['limits_z'] = self.limits_z
        
        if return_only:
            return self.params
        else:
            np.save(pathM+'motion_parameters.npy', self.params, allow_pickle = True)
    
class NonRigidLoad(NonRigidMeta):
    """
    Class that loads the non-rigid deformation (dr) using the motion_parameters.
    """
    def __init__(self, pathM):
        
        self.load(pathM)
        
        super().__init__()

    def load(self, pathM):
        """
        Loading the dictionary that contains all of they key parameters to generate dr.
        """
        self.params = np.load(pathM +'motion_parameters.npy', allow_pickle='TRUE').item()
        
        self.limits_x = self.params['limits_x']
        self.limits_y = self.params['limits_y']
        self.limits_z = self.params['limits_z']
        
        self.dx = MVPLoad(self.params['dx'], self.limits_x, self.limits_y, self.limits_z)
        if self.params['dy'] is None:  #When motion is in_plane
            self.dy = lambda pos: np.zeros(pos.shape[0])
        else:
            self.dy = MVPLoad(self.params['dy'], self.limits_x, self.limits_y, self.limits_z)
        self.dz = MVPLoad(self.params['dz'], self.limits_x, self.limits_y, self.limits_z)
        
        self.max_dx = self.params['max_dx']
        self.max_dy = self.params['max_dy']
        self.max_dz = self.params['max_dz']

        self.max_dr = self.params['max_dr']
#%% Unit tests
def _unit_test_collisions():
    x_lims = [20,40]
    min_sep = 0.34
    r = x_lims[0] * 1.1
    roots = np.array([r, r + min_sep * 2, r + min_sep * 5])
    
    candidate = roots[-1] + min_sep*0.5
    out = _no_collisions(c = candidate, roots = roots, min_sep=min_sep)
    assert out == False, f'_no_collisions failed. roots[-1] + min_sep*0.5 case. Should have been False, returned {out}. \n candidate = {candidate} \n roots = {roots} \n min_sep = {min_sep}'
    
    candidate = roots[-1] + min_sep*1.5
    out = _no_collisions(c = candidate, roots = roots, min_sep=min_sep)
    assert out == True, f'_no_collisions failed. roots[-1] + min_sep*1.5 case. Should have been True, returned {out}. \n candidate = {candidate} \n roots = {roots} \n min_sep = {min_sep}'

    candidate = roots[-1] + min_sep
    out = _no_collisions(c = candidate, roots = roots, min_sep=min_sep)
    assert out == True, f'_no_collisions failed. roots[-1] + min_sep*1.0 case. Should have been True, returned {out}. \n candidate = {candidate} \n roots = {roots} \n min_sep = {min_sep}'

    roots = np.array([r, r + min_sep * 2, r + min_sep * 5, r + min_sep * 6.1])
    candidate = r + min_sep * 6.0
    out = _no_collisions(c = candidate, roots = roots, min_sep=min_sep)
    assert out == False, f'_no_collisions failed. roots = [r+min_sep*5, r+min_sep*6.1], c = r+min_sep*6.0 case. Should have been False, returned {out}. \n candidate = {candidate} \n roots = {roots} \n min_sep = {min_sep}'
    
    out = _no_collisions(c = candidate, roots = np.array([]), min_sep=min_sep)
    assert out == True, f'_no_collisions failed. No roots case. Should have been True, returned {out}. \n candidate = {candidate} \n roots = {roots} \n min_sep = {min_sep}'
    
def _unit_test_lows_highs():
    x_lims = [20,40]
    min_sep = 0.34
    
    #Case of lower limit starting at x_lim[0] and r + min_sep * 3
    r = x_lims[0]
    roots = [r + min_sep, r + min_sep*2.1]
    lows, highs = _update_lows_highs(x_lims[0], x_lims[1], roots, min_sep)
    
    correct_answer = np.array([x_lims[0], r + min_sep*3.1])
    assert (lows == correct_answer).all(), f'_update_lows_highs failed. Returned lows = f{lows}. Expected lows = f{correct_answer}'
    correct_answer = np.array([r, x_lims[-1]])
    assert (highs == correct_answer).all(), f'_update_lows_highs failed. Returned highs = f{highs}. Expected highs = f{correct_answer}'

    #Case of r = [x_lims[0] + min_sep]
    roots = [x_lims[0] + min_sep]
    lows, highs = _update_lows_highs(x_lims[0], x_lims[1], roots, min_sep)
    
    correct_answer = np.array([x_lims[0], roots[0] + min_sep])
    assert (lows == correct_answer).all(), f'_update_lows_highs failed. Returned lows = f{lows}. Expected lows = f{correct_answer}'
    correct_answer = np.array([roots[0]-min_sep, x_lims[-1]])
    assert (highs == correct_answer).all(), f'_update_lows_highs failed. Returned highs = f{highs}. Expected highs = f{correct_answer}'
    
    #Case of r = [x_lims[0] + 0.5*min_sep]
    roots = [x_lims[0] + min_sep*0.5]
    lows, highs = _update_lows_highs(x_lims[0], x_lims[1], roots, min_sep)
    
    correct_answer = np.array([roots[0] + min_sep])
    assert (lows == correct_answer).all(), f'_update_lows_highs failed. Returned lows = f{lows}. Expected lows = f{correct_answer}'
    correct_answer = np.array([x_lims[-1]])
    assert (highs == correct_answer).all(), f'_update_lows_highs failed. Returned highs = f{highs}. Expected highs = f{correct_answer}'
    
    #Case of r = []
    roots = []
    lows, highs = _update_lows_highs(x_lims[0], x_lims[1], roots, min_sep)
    
    correct_answer = np.array([x_lims[0]])
    assert (lows == correct_answer).all(), f'_update_lows_highs failed. Returned lows = f{lows}. Expected lows = f{correct_answer}'
    correct_answer = np.array([x_lims[-1]])
    assert (highs == correct_answer).all(), f'_update_lows_highs failed. Returned highs = f{highs}. Expected highs = f{correct_answer}'

    #Case of roots = [x_lims[-1] - min_sep]
    roots = [x_lims[-1] - min_sep]
    lows, highs = _update_lows_highs(x_lims[0], x_lims[1], roots, min_sep)
    
    correct_answer = np.array([x_lims[0], x_lims[-1]])
    assert (lows == correct_answer).all(), f'_update_lows_highs failed. Returned lows = f{lows}. Expected lows = f{correct_answer}'
    correct_answer = np.array([roots[0] - min_sep, x_lims[-1]])
    assert (highs == correct_answer).all(), f'_update_lows_highs failed. Returned highs = f{highs}. Expected highs = f{correct_answer}'

    #Case of roots = [x_lims[-1]-0.5*min_sep]
    roots = [x_lims[-1] - 0.5*min_sep]
    lows, highs = _update_lows_highs(x_lims[0], x_lims[1], roots, min_sep)
    
    correct_answer = np.array([x_lims[0]])
    assert (lows == correct_answer).all(), f'_update_lows_highs failed. Returned lows = f{lows}. Expected lows = f{correct_answer}'
    correct_answer = np.array([roots[0]-min_sep])
    assert (highs == correct_answer).all(), f'_update_lows_highs failed. Returned highs = f{highs}. Expected highs = f{correct_answer}'

def _unit_test_remaining_space():
    x_lims = [20,30]
    min_sep = 0.34
    n_max = np.floor(abs(np.diff(x_lims)/min_sep)) + 1 # + 1 because of the open interval at edges
    x_min, x_max = x_lims
    r = x_min
    
    roots = [r]
    lows, highs = _update_lows_highs(x_min, x_max, roots, min_sep)
    out = _remaining_space(lows, highs, min_sep)
    assert(out==(n_max-1)), f'The expected remaining number of roots is {n_max-1}, the returned answer is {out}.'

    roots = [r + min_sep]
    lows, highs = _update_lows_highs(x_min, x_max, roots, min_sep)
    out = _remaining_space(lows, highs, min_sep)
    assert(out==(n_max-1)), f'The expected remaining number of roots is {n_max-1}, the returned answer is {out}.'

    roots = [r, r + min_sep]
    lows, highs = _update_lows_highs(x_min, x_max, roots, min_sep)
    out = _remaining_space(lows, highs, min_sep)
    assert(out==(n_max-2)), f'The expected remaining number of roots is {n_max-1}, the returned answer is {out}.'

    roots = [r, r + 2*min_sep]
    lows, highs = _update_lows_highs(x_min, x_max, roots, min_sep)
    out = _remaining_space(lows, highs, min_sep)
    assert(out==(n_max-2)), f'The expected remaining number of roots is {n_max-1}, the returned answer is {out}.'

    roots = [r, r + 1.9*min_sep]
    lows, highs = _update_lows_highs(x_min, x_max, roots, min_sep)
    out = _remaining_space(lows, highs, min_sep)
    assert(out==(n_max-3)), f'The expected remaining number of roots is {n_max-1}, the returned answer is {out}.'

def _unit_test_separated_roots():
    n = 0
    roots = _separated_roots(n, -10, 10, 3)
    assert (len(roots) == n), f'{n} roots were expected, {len(roots)} roots were generated'
    
    n = 3
    roots = _separated_roots(n, -10, 10, 3)
    assert (len(roots) == n), f'{n} roots were expected, {len(roots)} roots were generated'

    n = 4
    roots = _separated_roots(n, -10, 10, 3)
    assert (len(roots) == n), f'{n} roots were expected, {len(roots)} roots were generated'

    n = 5
    roots = _separated_roots(n, -10, 10, 3)
    assert (len(roots) == n), f'{n} roots were expected, {len(roots)} roots were generated'

    n = 6
    roots = _separated_roots(n, -10, 10, 3)
    assert (len(roots) == n), f'{n} roots were expected, {len(roots)} roots were generated'

    print('Testing maximum number of roots in space')
    n = 7 #maximum number of possible roots, given the limits
    roots = _separated_roots(n, -10, 10, 3)
    assert (len(roots) == n), f'{n} roots were expected, {len(roots)} roots were generated'
    print('Finished testing maximum number of roots in space')

def unit_test_roots():
    _unit_test_collisions()
    _unit_test_lows_highs()
    _unit_test_remaining_space()
    _unit_test_separated_roots()
    
    print('All tests of roots separated by min_sep passed.')

def unit_MVP(pos, limits_x, limits_y, limits_z, wavelength): #generates and loads MVP 
    encoded = MVPGenerate(limits_x, limits_y, limits_z, minimum_separation=wavelength)
    da_enc = encoded(pos)
    
    params = encoded.key_params()
    decoded = MVPLoad(params, limits_x, limits_y, limits_z)
    
    da_dec = decoded(pos)
    
    MVP_err = sum(abs(da_enc-da_dec))
    assert(np.isclose(MVP_err,0)), f'MVP error: {MVP_err}, should be 0.0'
    
    print('unit_MVP tests passed.')
    
def unit_dr(pos, limits_x, limits_y, limits_z, wavelength): #generates and loads dr
    pathM = './'
    max_dr = 17
    
    encoded = NonRigidGenerate(pathM, max_dr, limits_x, limits_y, limits_z, minimum_separation=wavelength)
    dr_enc = encoded(pos)
    
    decoded = NonRigidLoad(pathM)
    dr_dec = decoded(pos)
    
    numerical_dr = max(np.linalg.norm(dr_enc,axis=1))
    assert(abs(numerical_dr-max_dr) < max_dr*0.01), f'Maximum dr: {numerical_dr}, should be: {max_dr}'
    
    
    dr_err = sum(abs(dr_enc-dr_dec).flatten())
    assert(np.isclose(dr_err,0)), f'dr error: {dr_err}, should be 0.0'

    print('unit_dr tests passed.')

def unit_tests():
    #Encoded refers to encoded motion parameters
    #Decoded refers to decoded motion parameters
    
    x_lims = [-7,15]
    y_lims = [-5,6]
    z_lims = [30,60]
    wavelength = 1.33
    
    N = int(1e6) #number of scat
    pos = np.stack((np.random.uniform(*x_lims, N),np.random.uniform(*y_lims, N),np.random.uniform(*z_lims, N)),axis=1)
    
    unit_MVP(pos, x_lims, y_lims, z_lims, wavelength)
    
    unit_dr(pos, x_lims, y_lims, z_lims, wavelength)
#%%
if __name__ == '__main__':
    unit_test_roots()
    unit_tests()
    import os
    os.remove('./motion_parameters.npy')