from typing import NamedTuple

# import jax.numpy as np
import numpy as np

from gruyere.states import DesignState, PixelState, TouchState
from gruyere.viz import visualize


# Jax-JIT accept pytree (list / tuple / dict)
class Design(NamedTuple):
    # Reward
    reward: np.ndarray

    # Design
    x: np.ndarray

    # Void and Solid pixels
    p_v: np.ndarray
    p_s: np.ndarray

    # Void and Solid touches
    t_v: np.ndarray
    t_s: np.ndarray

    def _invert(self):
        # Invert the design
        # -x since DesignState set is -1/+1
        # x_void <--> x_solid
        return (
            Design(
                self.reward,
                -self.x,
                self.p_s,
                self.p_v,
                self.t_s,
                self.t_v
            )
        )

    def show(self):
        return visualize(self)


def _initialize_design(reward: np.ndarray) -> Design:
    shape = reward.shape

    # Design initialization
    ## Reward
    reward = reward

    ## Design
    x = np.ones(shape) * DesignState.UNASSIGNED

    ## Void and Solid pixels
    p_v = np.ones(shape) * PixelState.POSSIBLE
    p_s = np.ones(shape) * PixelState.POSSIBLE
    # p_v = np.ones(shape) * PixelState.IMPOSSIBLE
    # p_s = np.ones(shape) * PixelState.IMPOSSIBLE

    ## Void and Solid touches
    t_v = np.ones(shape) * TouchState.VALID
    t_s = np.ones(shape) * TouchState.VALID

    return Design(reward, x, p_v, p_s, t_v, t_s)
