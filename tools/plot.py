"""
Copyrights for original are Wookai's under MIT from:
https://github.com/Wookai/paper-tips-and-tricks/blob/master/src/python/plot_utils.py

Modifications are similarly under MIT, by Emil Lynegaard.

Utilities for plotting.
"""
import os
import subprocess
import tempfile
import matplotlib.pyplot as plt

LABEL_SIZE = 10
FONT_SIZE = 10
TICK_SIZE = 8
AXIS_LW = 0.6
PLOT_LW = 1.5


def new_figure():
    """
    Set all the sizes to the correct values and use
    tex fonts for all texts.
    """
    params = {
        "text.usetex": True,
        "figure.dpi": 200,
        "font.size": FONT_SIZE,
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.linewidth": AXIS_LW,
        "legend.fontsize": FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "font.family": "serif",
    }

    plt.rcParams.update(params)

    return plt.figure()


def save_figure(fig, file_name, fmt=None, dpi=300, tight=True):
    """
    Save a Matplotlib figure as EPS/PNG to the given path and trim it.
    """

    if not fmt:
        fmt = file_name.strip().split(".")[-1]

    if fmt not in ["eps", "png", "pdf"]:
        raise ValueError("unsupported format: %s" % (fmt,))

    extension = ".%s" % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(tmp_name, dpi=dpi)

    # trim it
    if fmt == "eps":
        subprocess.call(
            "epstool --bbox --copy %s %s" % (tmp_name, file_name), shell=True
        )
    elif fmt == "png":
        subprocess.call("convert %s -trim %s" % (tmp_name, file_name), shell=True)
