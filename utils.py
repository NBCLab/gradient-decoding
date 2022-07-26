"""Miscellaneous functions used for analyses."""
import os.path as op

import numpy as np
from matplotlib import pyplot as plt


def plot_dm_results(lambdas, output_dir):
    _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    ax1.set_xlabel("Component Nb")
    ax1.set_xlabel("Optimal Components")

    ax1.set_ylabel("Explained Variance Ratio")
    ax2.set_ylabel("Difference in Explained Variance Ratio")
    ax1.scatter(range(1, lambdas.size + 1, 1), (100 * lambdas) / lambdas.sum())
    gm_lambdas_diff = 100 * (lambdas[:-1] - lambdas[1:]) / lambdas.sum()
    ax2.scatter(range(1, lambdas.size, 1), gm_lambdas_diff)

    plt.savefig(op.join(output_dir, "lambdas.png"))
