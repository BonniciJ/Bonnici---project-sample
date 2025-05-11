
"""
Module containing pair creator function

V3:
V2 discontinued on 01/07/2022. The previous version can be found in the archieves
folder

"""
#%%
def _pairs(orig, frames_per_movie, return_inverse_order):
    """
    Returns: [[orig, orig], [orig, orig + 1], ... [orig, -1]]
    """
    
    indicies = range(frames_per_movie)
    pairs = [[indicies[orig], indicies[i+orig]] for i in range(frames_per_movie-orig)]
    if return_inverse_order:
        pairs += [[indicies[i+orig+1], indicies[orig]] for i in range(frames_per_movie-orig-1)]
    return pairs

def _pairs_separated(orig, frames_per_movie, return_inverse_order, sep):
    """
    Returns: [[orig, orig], [orig, orig + 1], ... [orig, orig + sep]]
    """
        
    indicies = range(frames_per_movie)
    pairs = [[indicies[orig], indicies[i+orig]] for i in range(sep+1)]
    if return_inverse_order:
        pairs += [[indicies[i+orig+1], indicies[orig]] for i in range(sep)]
    return pairs


#%%
def pairs_creator(pairing, frames_per_movie = 25, return_inverse_order = False):
    """
    Parameters
    ----------
    pairing: int/list/tuple
             int: provides all pairs starting from `pairing' up to `frames_per_movie':
                 [[int, int], [int, int+1], ... [int, frames_per_movie-1]], where int
                 is the `pairing'.
             
             list: same behavior as with int but for all values in the 
                 range of list.

             tuple: same behavior as with int but for each member in the tuple

    frames_per_movie: int
                      number of frames used per movie
    
    return_inverse_order: bool
        If True, also includes cases with inverse order [[m,n]] -> [[m,n], [n,m]]

    """
    if type(pairing) == int:
        return _pairs(pairing, frames_per_movie, return_inverse_order)
    elif type(pairing) == list:
        pairs = []
        for p in range(*pairing):
            pairs += _pairs(p, frames_per_movie, return_inverse_order)
        return pairs
    elif type(pairing) == tuple:
        pairs = []
        pairing = sorted(pairing)
        for p in pairing:
            pairs += _pairs(p, frames_per_movie, return_inverse_order)
        return pairs    
    
    elif type(pairing) == dict:
        max_sep = pairing['max_sep']
        starting = pairing['starting']
        
        if type(starting) == list: starting = range(starting[0], starting[-1] + 1) #if starting list, is a range of original startig samples
        if type(starting) == int: starting = [starting]
        
        assert max_sep <= frames_per_movie -1, 'Maximum separation cannot exceed number of frames-1.'
        
        pairs = []
        for og in starting:
            
            temp = max_sep + 1 + og
            if temp >= frames_per_movie: temp = frames_per_movie
            pairs += pairs_creator(og, temp, return_inverse_order)
            
        return pairs

