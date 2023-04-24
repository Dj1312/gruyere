from typing import Union

import numpy as onp
import jax.numpy as jnp
onp.random.seed(42)


def list_2id_to_1ids(ids: list[jnp.array, jnp.array]):
    # return list(
    #     zip(*map(lambda x: x.tolist(), ids))
    # )
    return zip(*map(lambda x: x.tolist(), ids))

x = onp.random.randint(-10, 10, size=(10, 10))
print(x)
print(x * (x > 0))

print(onp.where(x>0))

print(list_2id_to_1ids(onp.where(x>0)))
print(x[onp.where(x>0)])

# for ids in list_2id_to_1ids(onp.where(x>0)):
#     print(ids)

a = list_2id_to_1ids(onp.where(x>0))
print(list(a))
next(a)
print(list(a))
# print(list(next(list_2id_to_1ids(onp.where(x>0)))))
