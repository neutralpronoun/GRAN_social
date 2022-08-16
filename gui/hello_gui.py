import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd  #
import io

import pywebio
from pywebio.output import *
from pywebio.input import *

matplotlib.use("agg")


def dummy_sin(max_theta):
    x = np.linspace(0, max_theta * np.pi, 500)
    y = np.sin(x)

    fig, ax = plt.subplots(figsize = (12,6))

    ax.plot(x, y)

    buf = io.BytesIO()
    fig.savefig(buf)
    # put_image(buf.getvalue())
    return buf

# @use_scope("img", clear = True)
def update_plot(max_theta):
    with use_scope("img", clear = True):
        buf = dummy_sin(max_theta)
        put_image(buf.getvalue())

def sin_curve():
    with use_scope("sin_curve"):
        max_theta = slider(5, min_value = 0.001, max_value = 10, onchange=update_plot)


if __name__ == "__main__":
    sin_curve()
