import numpy as np

def mean_return(returns, k=4):
    """
    Compute mean return over last k periods per asset.
    If k > available time steps, use all available data.

    Parameters:
    -----------
    returns : np.ndarray
        Matrix of returns with shape (n_assets, n_periods)
    k : int
        Number of periods to consider from the end

    Returns:
    --------
    np.ndarray
        Array of mean returns per asset over last k periods, shape (n_assets,)
    """
    #print(returns.shape)
    n_periods = returns.shape[1]
    k_use = min(k, n_periods)
    returns_k = returns[:, -k_use:]  # slice last k_use columns
    return np.mean(returns_k, axis=1)


def historic_returns(returns, k=4):
    """
    Compute historic return over last k periods per asset,
    by multiplying (1 + r_i) and subtracting 1.
    If k > available time steps, use all available data.

    Parameters:
    -----------
    returns : np.ndarray
        Matrix of returns with shape (n_assets, n_periods)
    k : int
        Number of periods to consider from the end

    Returns:
    --------
    np.ndarray
        Array of historic returns per asset over last k periods, shape (n_assets,)
    """
    n_periods = returns.shape[1]
    k_use = min(k, n_periods)
    returns_k = returns[:, -k_use:]  # slice last k_use columns
    compounded = np.prod(1 + returns_k, axis=1) - 1
    return compounded


def sharpe_ratio(returns, k=4):
    """
    Compute Sharpe ratio over last k periods per asset.
    If k > available time steps, use all available data.

    Parameters:
    -----------
    returns : np.ndarray
        Matrix of returns with shape (n_assets, n_periods)
    k : int
        Number of periods to consider from the end

    Returns:
    --------
    np.ndarray
        Array of Sharpe ratios per asset over last k periods, shape (n_assets,)
        (mean divided by standard deviation of returns)
    """
    n_periods = returns.shape[1]
    k_use = min(k, n_periods)
    returns_k = returns[:, -k_use:]
    mean_ret = np.mean(returns_k, axis=1)
    std_ret = np.std(returns_k, axis=1, ddof=1)  # sample std dev
    sharpe = np.where(std_ret > 0, mean_ret / std_ret, 0)
    return sharpe
