from itertools import compress

import jax.numpy as jnp


def array_2id_to_1ids(ids: list[jnp.array, jnp.array]):
    return list(
        zip(*map(lambda x: x.tolist(), ids))
    )
    # return zip(*map(lambda x: x.tolist(), ids))


def from_list_id_couple_to_2tuples_ids(original_list):
    # return list(zip(*original_list))
    return tuple(zip(*original_list))


def from_2list_id_to_tuple_2ids(original_list):
    # return list(zip(*original_list))
    return tuple(zip(*original_list))


# More explanation here:
# https://stackoverflow.com/questions/10274774/python-elegant-and-efficient-ways-to-mask-a-list
def mask_list(original_list, bool_mask):
    return list(
        compress(original_list, bool_mask)
    )
