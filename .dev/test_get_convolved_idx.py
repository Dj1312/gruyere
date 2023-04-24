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


def get_convolved_idx(data, idxs0, brush):

    x_shape, y_shape = data.shape
    # Get the range from the brush POV centered on the pixel idxs0
    # which well be contained in the window
    brush_x_range = range(
        max(0, brush.shape[0] // 2 - idxs0[0]),
        min(brush.shape[0], brush.shape[0] // 2 + (x_shape - idxs0[0]))
    )
    brush_y_range = range(
        max(0, brush.shape[1] // 2 - idxs0[1]),
        min(brush.shape[1], brush.shape[1] // 2 + (y_shape - idxs0[1]))
    )
    # Get the range of the pixel with the brush centered at idxs0
    # which well be contained in the window
    x_range = range(
        max(0, idxs0[0] - brush.shape[0] // 2),
        min(idxs0[0] + 1 + brush.shape[0] // 2, x_shape)
    )
    y_range = range(
        max(0, idxs0[1] - brush.shape[1] // 2),
        min(idxs0[1] + 1 + brush.shape[1] // 2, y_shape)
    )

    # Generate the list of idx
    idx_brush_window = list(product(brush_x_range, brush_y_range))
    print(idx_brush_window)
    mask_brush = (brush[*list(zip(*idx_brush_window))] == True)
    idx_data_window = list(product(list(x_range), list(y_range)))
    print(max(0, idxs0[0] - brush.shape[0] // 2),
        min(idxs0[0] + brush.shape[0] // 2, x_shape))
    print(max(0, idxs0[1] - brush.shape[1] // 2),
        min(idxs0[1] + brush.shape[1] // 2, y_shape))
    print(idx_data_window)
    idxs_conv = list(compress(idx_data_window,mask_brush))

    return idxs_conv


idxs0 = np.array([0,6])
brush = circular_brush(5)
print(brush)

window_shape = [6, 8]
data = np.zeros(window_shape)

idxs_final = get_convolved_idx(data, idxs0, brush)

# data[*zip(*idxs_final)] = 3
data[*list(zip(*idxs_final))] = 3

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(data)
plt.savefig('test_convolved.png')

print(idxs_final)
print(*list(zip(*idxs_final)))

list_2ids = list(zip(*idxs_final))
print(list(zip(*list_2ids)) == idxs_final)

print(type(list(zip(*idxs_final))[0]))
