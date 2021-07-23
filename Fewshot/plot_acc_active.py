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




acc_cifar_conv_1shot = [36.38, 69.95, 70.30, 70.15, 70.65, 70.66, 70.00, 70.01, 70.74, 70.56]
acc_cifar_conv_5shot = [68.72, 80.45, 80.37, 80.46, 80.77, 80.68, 80.52, 80.55, 80.64, 80.39]
acc_mini_conv_1shot = [25.543, 61.997, 61.927, 61.591, 61.389, 61.600, 61.681, 61.809, 61.813, 61.892]
acc_mini_conv_5shot = [62.63, 74.62, 74.54, 74.54, 74.62, 74.67, 74.67, 74.67, 74.63, 74.79]
acc_cifar_wrn_1shot = [79.21, 79.88, 79.53, 79.19, 79.53, 79.19, 79.42, 79.89, 79.659, 80.01]
acc_cifar_wrn_5shot = [81.03, 86.90, 86.25, 86.26, 86.68, 86.73, 86.83, 87.08, 86.88, 87.02]
acc_mini_wrn_1shot = [61.005, 69.976, 70.030, 69.759, 69.926, 70.241, 69.868, 69.797, 69.579, 70.094,]
acc_mini_wrn_5shot = [73.84, 81.89, 82.10, 82.30, 82.26, 82.15, 82.08, 82.52, 82.30, 82.29]

act_num = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]

plt.figure(0, figsize=set_size(width * 0.5))
plt.plot(act_num, acc_cifar_conv_1shot - np.mean(acc_cifar_conv_1shot), marker='o', label="Conv-4-64, CIFAR-FS, 1-shot")
plt.plot(act_num, acc_cifar_conv_5shot - np.mean(acc_cifar_conv_5shot), marker='o', label="Conv-4-64, CIFAR-FS, 5-shot")
plt.plot(act_num, acc_mini_conv_1shot - np.mean(acc_mini_conv_1shot), marker='o', label="Conv-4-64, MiniImageNet, 1-shot")
plt.plot(act_num, acc_mini_conv_5shot - np.mean(acc_mini_conv_5shot), marker='o', label="Conv-4-64, MiniImageNet, 5-shot")
plt.plot(act_num, acc_cifar_wrn_1shot - np.mean(acc_cifar_wrn_1shot), marker='o', label="WRN-28-10, CIFAR-FS, 1-shot")
plt.plot(act_num, acc_cifar_wrn_5shot - np.mean(acc_cifar_wrn_5shot), marker='o', label="WRN-28-10, CIFAR-FS, 5-shot")
plt.plot(act_num, acc_mini_wrn_1shot - np.mean(acc_mini_wrn_1shot), marker='o', label="WRN-28-10, MiniImageNet, 1-shot")
plt.plot(act_num, acc_mini_wrn_5shot - np.mean(acc_mini_wrn_5shot), marker='o', label="WRN-28-10, MiniImageNet, 5-shot")
plt.xlabel(r"Active mechanism count, $K$")
plt.ylabel(r"Accuracy$-\frac{\sum \textrm{Accuracy}}{|\textrm{Accuracy}|}$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('act.pdf')
plt.show()