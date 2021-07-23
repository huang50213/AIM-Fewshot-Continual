import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "Times New Roman",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 4.5,
        # "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1,
        "lines.markersize": 2,
}

mpl.rcParams.update(nice_fonts)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# sns.set_context("paper", rc=nice_fonts)

width = 430.00462

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 0.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim




acc_cifar_conv_1shot = [70.80, 71.09, 70.68, 69.95, 70.01, 68.82, 67.79, 66.90]
acc_cifar_conv_5shot = [80.50, 80.48, 80.36, 79.96, 79.60, 79.21, 78.90, 78.35]
acc_mini_conv_1shot = [61.90, 61.90, 62.13, 61.74, 61.70, 60.73, 60.34, 59.43]
acc_mini_conv_5shot = [74.55, 74.55, 74.66, 74.24, 74.12, 73.58, 73.19, 72.89]
acc_cifar_wrn_1shot = [79.19, 80.20, 80.20, 79.95, 80.17, 80.18, 80.33, 80.13]
acc_cifar_wrn_5shot = [87.04, 87.34, 87.19, 87.10, 87.07, 86.68, 86.63, 86.22]
acc_mini_wrn_1shot = [71.03, 71.22, 71.08, 70.57, 70.38, 70.20, 70.40, 69.34]
acc_mini_wrn_5shot = [82.30, 82.25, 82.25, 82.25, 81.86, 81.65, 81.60, 81.19]

sto_num = [8, 10, 12, 16, 20, 24, 28, 32]

plt.figure(0, figsize=set_size(width * 0.5))
plt.plot(sto_num, acc_cifar_conv_1shot - np.mean(acc_cifar_conv_1shot), marker='o', label="Conv-4-64, CIFAR-FS, 1-shot")
plt.plot(sto_num, acc_cifar_conv_5shot - np.mean(acc_cifar_conv_5shot), marker='o', label="Conv-4-64, CIFAR-FS, 5-shot")
plt.plot(sto_num, acc_mini_conv_1shot - np.mean(acc_mini_conv_1shot), marker='o', label="Conv-4-64, MiniImageNet, 1-shot")
plt.plot(sto_num, acc_mini_conv_5shot - np.mean(acc_mini_conv_5shot), marker='o', label="Conv-4-64, MiniImageNet, 5-shot")
plt.plot(sto_num, acc_cifar_wrn_1shot - np.mean(acc_cifar_wrn_1shot), marker='o', label="WRN-28-10, CIFAR-FS, 1-shot")
plt.plot(sto_num, acc_cifar_wrn_5shot - np.mean(acc_cifar_wrn_5shot), marker='o', label="WRN-28-10, CIFAR-FS, 5-shot")
plt.plot(sto_num, acc_mini_wrn_1shot - np.mean(acc_mini_wrn_1shot), marker='o', label="WRN-28-10, MiniImageNet, 1-shot")
plt.plot(sto_num, acc_mini_wrn_5shot - np.mean(acc_mini_wrn_5shot), marker='o', label="WRN-28-10, MiniImageNet, 5-shot")
plt.xlabel(r"Stochastic sampling count, $K+l$")
plt.ylabel(r"Accuracy$-\frac{\sum \textrm{Accuracy}}{|\textrm{Accuracy}|}$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('sto.pdf')
plt.show()