from typing import TYPE_CHECKING
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from gruyere.states import TouchState, PixelState, DesignState


if TYPE_CHECKING:
    from gruyere.design import Design


# Original cpde from the work of Floris Laporte
# https://github.com/flaport/inverse_design/blob/master/inverse_design/design.py


def visualize(des: 'Design'):
    nx, ny = des.x.shape
    _, axs = plt.subplots(1, 5, figsize=(15, 3 * nx / ny))

    data_dict = {'Design': des.x,
                 'Void pixels': des.p_v,
                 'Solid pixels': des.p_s,
                 'Void touches': des.t_v,
                 'Solid touches': des.t_s,
                 }
    for i, (k, v) in enumerate(data_dict.items()):
        plot_pixel_array(des, axs[i], k, v)


def plot_pixel_array(des: 'Design', ax, name, values):
    if name == 'Design':
        _cmap = ListedColormap(
            list({DesignState.VOID: "#cbcbcb",
                  DesignState.UNASSIGNED: "#929292",
                  DesignState.SOLID: "#515151",
                  DesignState.TEST: "#ffffff",}.values()))
        _bounds = list((DesignState.VOID,
                        DesignState.UNASSIGNED,
                        DesignState.SOLID,
                        DesignState.TEST))

    elif name in ['Void pixels', 'Solid pixels']:
        _cmap = ListedColormap(
            list({PixelState.IMPOSSIBLE: "#8dd3c7",
                  PixelState.EXISTING: "#ffffb3",
                  PixelState.POSSIBLE: "#bebada",
                  PixelState.REQUIRED: "#fb7f72",
                  PixelState.TEST: "#ffffff",}.values()))
        _bounds = list((PixelState.IMPOSSIBLE,
                        PixelState.EXISTING,
                        PixelState.POSSIBLE,
                        PixelState.REQUIRED,
                        PixelState.TEST))

    elif name in ['Void touches', 'Solid touches']:
        _cmap = ListedColormap(
            list({TouchState.EMPTY: "#00ff00",
                  TouchState.INVALID: "#7fb1d3",
                  TouchState.EXISTING: "#fdb462",
                  TouchState.VALID: "#b3de69",
                  TouchState.FREE: "#fccde5",
                  TouchState.RESOLVING: "#e0e0e0",
                  TouchState.TEST: "#ffffff"}.values()))
        _bounds = list((TouchState.EMPTY,
                        TouchState.INVALID,
                        TouchState.EXISTING,
                        TouchState.VALID,
                        TouchState.FREE,
                        TouchState.RESOLVING,
                        TouchState.TEST))

    else:
        raise ValueError

    _norm = mpl.colors.BoundaryNorm(_bounds, _cmap.N - 1)
    nx, ny = values.shape
    ax.set_title(name)
    ax.imshow(values, interpolation='none', cmap=_cmap, norm=_norm)

    ax.set_yticks(np.arange(nx) + 0.5)
    ax.set_yticklabels(["" for i in range(nx)])
    ax.set_xticks(np.arange(ny) + 0.5)
    ax.set_xticklabels(["" for i in range(ny)])
    ax.set_yticks(np.arange(nx), minor=True)

    ax.set_yticklabels([f"{chr(65 + i)}" for i in range(nx)], minor=True)
    ax.set_xticks(np.arange(ny), minor=True)
    ax.set_xticklabels([f"{i}" for i in range(ny)], minor=True)
    ax.set_xlim(-0.5, ny - 0.5)
    ax.set_ylim(nx - 0.5, -0.5)
    ax.grid(visible=True, which="major", c="k")
