from scipy.stats import weightedtau
import numpy as np

def aggressive_rank_weight(rank, alpha=0.0):

    """
    Create exponentially decaying weights that heavily favor top ranks.
    
    Parameters:
    -----------
    rank : int
        The rank position (0-indexed)
    alpha : float
        Controls the decay speed - higher values create more aggressive weighting
    
    Returns:
    --------
    float
        The weight value, decaying exponentially with rank
    """
    #return np.exp(-alpha * rank)
    return 1/(1+rank)

# Custom weighted Kendall tau function using our aggressive weighting
def aggressive_weightedtau(x, y):
    return weightedtau(x, y, weigher=aggressive_rank_weight, additive=False)