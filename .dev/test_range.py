import numpy as np
import jax.numpy as jnp
from itertools import product, filterfalse

# Global flag to set a specific platform, must be used at startup.
import jax
jax.config.update('jax_platform_name', 'cpu')

x_range = range(-5, 12)
y_range = range(-3, 6)

brush = np.random.randint(0, 10, size=(len(x_range), len(y_range)))

idx = list(product(
    *map(lambda arr: [idx for idx in arr if idx >= 0], [x_range, y_range])
))

tidx = product(
    *map(lambda arr: (idx for idx in arr if idx >= 0), [x_range, y_range])
)

tab = jnp.zeros((max(x_range) + 2, max(y_range) + 2))
print(idx)
print(tuple(idx))
print(tab[tuple(zip(*idx))])
print(list(zip(*idx)))
print('\n')
print(tuple(zip(*map(lambda arr: [idx for idx in arr if idx >= 0], [x_range, y_range]))))
exit()
print(list(tidx))
exit()
print(list(idx))
print(tuple(tidx))

tab = jnp.zeros((max(x_range) + 2, max(y_range) + 2))
# tab = tab.at[tuple(tidx)].set(1)
tab = tab.at[list(tidx)].set(1)
print(tab)
print(tuple(idx))
