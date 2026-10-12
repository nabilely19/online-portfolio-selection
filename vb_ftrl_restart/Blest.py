import numpy as np


def blest_rank_corr(R, S):
    """
    Raw sample statistic for Blest's rank correlation.

    Parameters
    ----------
    R : array-like
        Numerical ranking of the first list (integers from 1 to n).
    S : array-like
        Numerical ranking of the second list (integers from 1 to n).

    Returns
    -------
    float
        Value of gamma_n (may exceed [-1,1] for finite samples).
    """
    R = np.asarray(R, dtype=float)
    S = np.asarray(S, dtype=float)
    n = R.size
    if n < 2:
        raise ValueError("At least two elements are required to compute the correlation.")
    return (2 * n + 1) / (n - 1) - (12 / (n**2 - n)) * np.sum((1 - R / (n + 1))**2 * S)



def blest_coeff(rank_labels1, rank_labels2):
    """
    Compute the normalized Blest rank correlation from two ordered label lists.

    Parameters
    ----------
    rank_labels1 : sequence
        List of labels (e.g., tickers) ordered by their ranking descending.
    rank_labels2 : sequence
        Second list of the same labels in their ranking order.

    Returns
    -------
    float
        Normalized Blest rank correlation in [-1, 1].
    """
    if len(rank_labels1) != len(rank_labels2):
        raise ValueError("Both lists must have the same length.")
    if set(rank_labels1) != set(rank_labels2):
        raise ValueError("Both lists must contain the same elements.")

    # Build mapping from label → numeric rank based on the first list
    rank_map = {label: idx + 1 for idx, label in enumerate(rank_labels1)}

    # Create numeric rank arrays for both lists using that mapping
    R = np.array([rank_map[label] for label in rank_labels1], dtype=float)
    S = np.array([rank_map[label] for label in rank_labels2], dtype=float)

    # Optional: inspect the numeric ranks
    #print("R", R)
    #print("S", S)

    return blest_rank_corr(R, S)
