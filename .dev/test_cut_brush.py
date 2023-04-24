import numpy as np
import numpy as onp
import matplotlib.pyplot as plt

from itertools import product, compress

def circular_brush(diameter):
    radius = diameter / 2
    X, Y = onp.mgrid[-radius:radius:1j * diameter, -radius:radius:1j * diameter]
    _int = lambda x: onp.array(x, dtype=int)
    brush = _int(X) ** 2 + _int(Y) ** 2 < radius ** 2
    return brush


idxs0 = np.array([5,95])
brush = circular_brush(20)

# brush_centered = brush[
#     max(0, idxs0[0] - brush.shape[0] // 2):min(brush.shape[0], idxs0[0] + brush.shape[0] // 2),
#     max(0, idxs0[1] - brush.shape[1] // 2):min(brush.shape[1], idxs0[1] + brush.shape[1] // 2)
# ]
window_shape = [200, 100]

print(
    idxs0[0] + brush.shape[0] // 2, brush.shape[0],
    '\n',
    idxs0[1] + brush.shape[1] // 2, brush.shape[1],
)

print(brush.shape[0] // 2 - idxs0[0])
print(max(0, brush.shape[0] // 2 - idxs0[0]))

print(brush.shape[0] // 2 + (window_shape[0] - idxs0[0]))
print(max(0, brush.shape[0] // 2 - idxs0[0]),min(brush.shape[0], brush.shape[0] // 2 + (window_shape[0] - idxs0[0])))
print(max(0, brush.shape[1] // 2 - idxs0[1]),min(brush.shape[1], brush.shape[1] // 2 + (window_shape[1] - idxs0[1])))

brush_centered = brush[
    max(0, brush.shape[0] // 2 - idxs0[0]):min(brush.shape[0], brush.shape[0] // 2 + (window_shape[0] - idxs0[0])),
    max(0, brush.shape[1] // 2 - idxs0[1]):min(brush.shape[1], brush.shape[1] // 2 + (window_shape[1] - idxs0[1])),
]

# brush_centered = brush[
#     max(0, idxs0[0] + brush.shape[0] // 2):brush.shape[0],min(brush.shape[0], idxs0[0] - brush.shape[0] // 2),
#     max(0, idxs0[1] + brush.shape[1] // 2):brush.shape[1]#min(brush.shape[1], idxs0[1] - brush.shape[1] // 2)
# ]

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(brush, origin='lower')
# ax[1].imshow(brush_centered, origin='lower')
# plt.savefig('test')


x_shape, y_shape = window_shape
x_range = range(
    max(0, brush.shape[0] // 2 - idxs0[0]),
    min(brush.shape[0], brush.shape[0] // 2 + (x_shape - idxs0[0]))
)
y_range = range(
    max(0, brush.shape[1] // 2 - idxs0[1]),
    min(brush.shape[1], brush.shape[1] // 2 + (y_shape - idxs0[1]))
)
print(x_range)
print(list(product(list(x_range), list(y_range))))
# idx_brush_window = list(product(list(x_range), list(y_range)))
idx_brush_window = list(product(x_range, y_range))

print(idx_brush_window)
print(*list(zip(*idx_brush_window)))
print(brush[*list(zip(*idx_brush_window))].shape)

# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(brush, origin='lower')
# ax[1].imshow(brush_centered, origin='lower')
# ax[2].imshow(brush[*list(zip(*tst))].reshape(brush.shape), origin='lower')
# plt.savefig('test')
idx_brush = product(list(x_range), list(y_range))
bool_brush = brush[*list(zip(*idx_brush_window))] == True

print(brush[*list(zip(*idx_brush_window))])
idx_keep = list(compress(idx_brush, bool_brush))
print(brush[*list(zip(*idx_keep))])

# sset =
print(bool_brush)
print(idx_keep)
# print(set(idx_brush).union(set(bool_brush)))
print(brush[*list(zip(*idx_keep))])
