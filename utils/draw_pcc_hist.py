# @Author : CyIce
# @Time : 2024/7/3 9:21

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# To draw the histogram of the delta PCC values.


def remove_outliers(data, m=3):
    mean = np.mean(data)
    std = np.std(data)
    # Keep only the data within m standard deviations from the mean
    filtered_data = data[np.abs(data - mean) < m * std]
    return filtered_data


def draw_hist(delta, title, bins=100, sample_size=None, saved_path=None):
    plt.clf()
    delta_flattened = delta.flatten()
    # Filter out non-finite values
    delta_flattened = delta_flattened[np.isfinite(delta_flattened)]
    # Remove outliers
    delta_flattened = remove_outliers(delta_flattened, m=3)
    if sample_size is not None:
        sample_size = int(sample_size)
        if sample_size > len(delta_flattened):
            sample_size = len(delta_flattened)
        # Sample the specified number of data points
        delta_flattened = np.random.choice(delta_flattened, sample_size, replace=False)

    # Draw frequency distribution histogram
    plt.hist(delta_flattened, bins=bins, edgecolor='k', alpha=0.7)

    # Calculate the standard normal distribution curve
    mu, std = norm.fit(delta_flattened)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    # Draw the normal distribution curve in red dashed line
    plt.plot(x, p, 'r--', linewidth=2)

    plt.title(title)
    plt.xlabel('Delta PCC')
    plt.ylabel('Density')
    plt.grid(False)
    if saved_path is not None:
        plt.savefig(f"{saved_path}/{title}.png", dpi=300)
    else:
        plt.show()
