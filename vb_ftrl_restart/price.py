import numpy as np
import numpy.random as rd

def returns(n,T):
    """n : nombre d'actifs; T: taille de la timeserie"""
    mu = rd.normal(0, 0.001, n)
    mu = rd.normal(mu,0.0001,(T,n))

    sigma = rd.normal(0,0.01,(T,n))  # l'écart type est typiquement entre 0.005 et 0.01
    sigma = np.exp(sigma)
    sigma = np.multiply.accumulate(sigma,0)*0.005
    logReturn = rd.normal(mu,sigma,(T,n))
    Return = np.exp(logReturn)
    return Return


def prix(n,T,alpha=0.005):
    """n : nombre d'actifs; T: taille de la timeserie"""
    mu = np.random.normal(0, 0.00001, (T,n))
    alpha = np.multiply.accumulate(np.exp(rd.normal(0,0.01,(T,n))),0)*0.01
    logReturn = np.random.normal(mu,alpha,(T,n))
    Return = np.exp(logReturn)
    # Stock = np.multiply.accumulate(Return,0)
    return Return # Stock*100

def rescale(returns, method = "sigmoid",threshold = 0.05):
    if method == "sigmoid":
        return 1 / (1 + np.exp(-returns*40)) - 0.5
    if method =="minmax":
        maxi = threshold
        scaled = (returns)/(maxi)
        scaled[scaled > 1] = 1
        scaled[scaled < -1 ] = -1
        return (scaled)*10