import numpy as np

# Initialize a single, efficient random number generator
_rng = np.random.default_rng()

def generate_returns(n: int, T: int) -> np.ndarray:
    """
    Simulate asset returns over time.

    Parameters
    ----------
    n : int
        Number of assets.
    T : int
        Length of the time series.

    Returns
    -------
    returns : np.ndarray
        An array of shape (T, n) with simulated multiplicative returns.
    """
    # Base drifts: shape (n,) broadcasted to (T, n)
    mu0 = _rng.normal(0, 0.001, size=n)
    mu = _rng.normal(loc=np.broadcast_to(mu0, (T, n)), scale=0.0001)

    # Volatility process via cumprod of log-normal increments
    vol_increments = np.exp(_rng.normal(0, 0.01, size=(T, n)))
    sigma = 0.005 * np.cumprod(vol_increments, axis=0)

    # Draw log-returns and convert to multiplicative returns
    log_r = _rng.normal(loc=mu, scale=sigma)
    return np.exp(log_r)


def generate_prices(n: int, T: int) -> np.ndarray:
    """
    Simulate asset price returns over time with a small drift.

    Parameters
    ----------
    n : int
        Number of assets.
    T : int
        Length of the time series.

    Returns
    -------
    prices : np.ndarray
        An array of shape (T, n) with simulated multiplicative returns.
    """
    mu = _rng.normal(0, 1e-5, size=(T, n))
    alpha_increments = np.exp(_rng.normal(0, 0.01, size=(T, n)))
    alpha = 0.01 * np.cumprod(alpha_increments, axis=0)

    log_r = _rng.normal(loc=mu, scale=alpha)
    return np.exp(log_r)


def rescale(x: np.ndarray, method: str = "sigmoid", threshold: float = 0.05) -> np.ndarray:
    """
    Rescale an array of returns.

    Parameters
    ----------
    x : np.ndarray
        Input array of returns.
    method : str
        'sigmoid' for a smooth transform, 'minmax' for clipping.
    threshold : float
        Max absolute value for minmax scaling.

    Returns
    -------
    np.ndarray
        Rescaled values.
    """
    if method == "sigmoid":
        # Fast sigmoid transform
        return 1 / (1 + np.exp(-40 * x)) - 0.5
    elif method == "minmax":
        scaled = x / threshold
        # In-place clipping for speed
        np.clip(scaled, -1, 1, out=scaled)
        return scaled * 10
    else:
        raise ValueError(f"Unknown rescaling method: {method}")


'''tests
n=100, T=1000
Fonction | Ancienne (s) | Optimisée (s) | Facteur de speed-up
returns | 0 .0372 | 0 .0260 | × 1 .43
prix | 0 .0332 | 0 .0230 | × 1 .44
rescale_sigmoid | 0 .0010 | 0 .0009 | × 1 .10
rescale_minmax | 0 .0051 | 0 .0010 | × 4 .95
'''