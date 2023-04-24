from typing import NamedTuple

import jax.numpy as jnp
# import numpy as jnp

from gruyere.states import DesignState, PixelState, TouchState
from gruyere.viz import visualize


# Jax-JIT accept pytree (list / tuple / dict)
class Design(NamedTuple):
    # Reward
    reward: jnp.ndarray

    # Design
    x: jnp.ndarray

    # Void and Solid pixels
    p_v: jnp.ndarray
    p_s: jnp.ndarray

    # Void and Solid touches
    t_v: jnp.ndarray
    t_s: jnp.ndarray

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


def _initialize_design(reward: jnp.ndarray) -> Design:
    shape = reward.shape

    # Design initialization
    ## Reward
    reward = reward

    ## Design
    x = jnp.ones(shape) * DesignState.UNASSIGNED

    ## Void and Solid pixels
    p_v = jnp.ones(shape) * PixelState.POSSIBLE
    p_s = jnp.ones(shape) * PixelState.POSSIBLE
    # p_v = jnp.ones(shape) * PixelState.IMPOSSIBLE
    # p_s = jnp.ones(shape) * PixelState.IMPOSSIBLE

    ## Void and Solid touches
    t_v = jnp.ones(shape) * TouchState.VALID
    t_s = jnp.ones(shape) * TouchState.VALID

    return Design(reward, x, p_v, p_s, t_v, t_s)
