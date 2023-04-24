from typing import Union

import numpy as onp
import jax.numpy as jnp
onp.random.seed(42)


def list_2id_to_1ids(ids: list[jnp.array, jnp.array]):
    return list(
        zip(*map(lambda x: x.tolist(), ids))
    )
    # return zip(*map(lambda x: x.tolist(), ids))

x = onp.random.randint(-10, 10, size=(10, 10))
print(x)
print(list_2id_to_1ids(onp.where(x>0)))

# xjax = jnp.empty_like(x)
xjax = jnp.zeros(x.shape)
# xjax = xjax.at[list_2id_to_1ids(onp.where(x>0))].set(1)
xjax = xjax.at[onp.where(x>0)].set(1)
print(x > 0)
print(xjax)
