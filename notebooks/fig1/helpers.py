"""
Helper functions for the Fig. 1 notebooks.
"""
import numpy as np


# probability of switch
def get_proportions(X, y, x_bins):
    bin_edges = x_bins
    
    proportions = []
    bin_counts = []

    for k in range(len(bin_edges) - 1):
        mask = (X >= bin_edges[k]) & (X < bin_edges[k+1])
        if k == len(bin_edges) - 2:  # Include the last edge
            mask = (X >= bin_edges[k]) & (X <= bin_edges[k+1])

        if np.sum(mask) > 0:  # Avoid division by zero
            prop_correct = np.mean(y[mask])
            proportions.append(prop_correct)
            bin_counts.append(np.sum(mask))
        else:
            proportions.append(np.nan)
            bin_counts.append(0)

    return np.array(proportions)
