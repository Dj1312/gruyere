import numpy as np
import jax

from scipy.signal import convolve2d
# import scipy
from functools import partial
from itertools import compress

from .design import Design, _initialize_design
from .misc import array_2id_to_1ids, mask_list
from .ops import convolve_pixel
from .states import DesignState, FreeState, PixelState, TouchState
from .touches import add_solid_touches, add_void_touches


# Generator received the reward post transform with a brush
# @partial(jax.custom_jvp, nondiff_argnums=(1,))
def generator(reward, brush):
    # total_reward = convolve2d(reward, brush, mode='same')
    brush_convolved = (convolve2d(brush * 1.0, brush) != 0.0)

    # Sort by abs value
    # -> If negative: high probability of hole
    # -> If positive: high probability of solid
    # TODO: Study this
    # total_reward = jsp.signal.convolve2d(reward, brush, mode='same')
    total_reward = reward

    order = np.argsort(np.abs(total_reward).flatten())
    id_full_sorted = list(
        zip(
            *map(lambda x: x.tolist(), np.unravel_index(order, reward.shape))
        )
    )
    # id_sorted = id_full_sorted

    # Initialize designzz
    # initialize empty t^s_b and t^v_b
    # feas_design = Design(reward.shape)
    # feas_design = Design(reward)
    # feas_design = _initialize_design(reward.shape)
    feas_design = _initialize_design(reward)

    # while design is incomplete do
    while (feas_design.x == DesignState.UNASSIGNED).any():
        # Start from the highest reward
        # feas_design = _step_generator(
        #     feas_design, brush, id_sorted[-1], total_reward
        # )
        bool_touch_exist = (
            (feas_design.t_v.flatten() == TouchState.VALID)
            | (feas_design.t_s.flatten() == TouchState.VALID)
            | (feas_design.t_v.flatten() == TouchState.FREE)
            | (feas_design.t_s.flatten() == TouchState.FREE)
        )
        id_sorted = mask_list(
            id_full_sorted, bool_touch_exist[order]
        )
        feas_design = _step_generator(
            feas_design, brush, brush_convolved, id_sorted[-1]
        )

    return feas_design


# TODO: Make another generator that take also reward and update it
def _step_generator(
    # des: Design, brush: np.array, id: tuple[int, int], rew: np.array
    des: Design, brush: np.array, brush_convolved: np.array, id: tuple[int, int]
) -> Design:
    # if np.any(DesignState.is_solid(des.x[idx])):
    #     # If the pixel is solid, then the touch is solid
    #     des = _update_touch(des, idx, TouchState.SOLID)

    # if free touches exist then
    if np.any(des.t_v == TouchState.FREE):
        # updated_des = update_free_touches(des, FreeState.VOID)
        # # idxs = choose_valid_touch(des.reward, des.t_v == TouchState.FREE)
        idxs = array_2id_to_1ids(np.where(des.t_v == TouchState.FREE))
        updated_des = add_void_touches(des, idxs, brush, brush_convolved)
    elif np.any(des.t_s == TouchState.FREE):
        # updated_des = update_free_touches(des, FreeState.SOLID)
        # # idxs = choose_valid_touch(des.reward, des.t_s == TouchState.FREE)
        idxs = array_2id_to_1ids(np.where(des.t_s == TouchState.FREE))
        updated_des = add_solid_touches(des, idxs, brush, brush_convolved)

    # else if resolving touches exist then
    #   select a single resolving touch
    elif np.any(des.p_v == PixelState.REQUIRED):
        mask_req_void = (des.p_v == PixelState.REQUIRED)
        id_next = choose_higher_masked_val(des.reward, mask_req_void)
        updated_des = add_void_touches(des, [id_next], brush, brush_convolved)
    elif np.any(des.p_s == PixelState.REQUIRED):
        mask_req_solid = (des.p_s == PixelState.REQUIRED)
        id_next = choose_higher_masked_val(des.reward, mask_req_solid)
        updated_des = add_solid_touches(des, [id_next], brush, brush_convolved)

    else:
        # select a single valid touch
        if des.reward[id] > 0:
            updated_des = add_solid_touches(des, [id], brush, brush_convolved)
        else:
            updated_des = add_void_touches(des, [id], brush, brush_convolved)

    return updated_des


# # TODO: Test the solution of Jax documentation
# # https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html
# @generator.defjvp
def conditional_generator_jvp(brush, primals, tangents):
    reward, = primals
    # The gradient of a non-differentiable [is replaced] with that of an estimator
    # A typical estimator [...] is the identity operation.
    # --> The gradient of the identity will be implemented by returning the tangent
    reward_dot, = tangents
    return generator(brush, reward), reward_dot


# The next touch is those from the reward with the highest value
# @jax.jit
# TODO: Take only the idx in account
def choose_higher_masked_val(array, mask):
    idx_max = np.abs((array * mask).flatten()).argmax()
    return np.unravel_index(idx_max, shape=array.shape)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import brushes
    np.random.seed(42)
    brush = brushes.notched_square_brush(13, 1)

    L = 100
    rew = np.random.rand(L,L)

    beta = 2
    post_transform = np.tanh(beta * convolve2d(rew, brush, mode='same'))

    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(rew.T, origin='lower')
    ax[1].imshow(post_transform.T, origin='lower')
    ax[2].imshow(generator(rew, brush).T, origin='lower')
    plt.show()
