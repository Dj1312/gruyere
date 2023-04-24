from itertools import compress, product
import numpy as np

# from gruyere.misc import from_2list_id_to_list_2ids, from_list_2ids_to_2list_id
# from gruyere.misc import from_list_2ids_to_2tuples_id, from_2list_id_to_tuple_2ids


def convolve_pixel(data, coords_x, coords_y, brush):
    """
    Convolve only 1 pixel of the array "data" with the 2nd array "brush"
    at coordinates (coords_x, coords_y)
    """
    # Get the size of the brush
    brush_size_x, brush_size_y = brush.shape

    # Get the coordinates of the pixel to convolve
    x, y = coords_x, coords_y

    # Get the range of indices to apply the brush on
    x_range = slice(max(0, x - brush_size_x // 2), min(data.shape[0], x + brush_size_x // 2 + 1))
    y_range = slice(max(0, y - brush_size_y // 2), min(data.shape[1], y + brush_size_y // 2 + 1))

    # Apply the brush to the pixel
    data[x_range, y_range] += (
        data[x, y] * brush[
            max(0, brush_size_x // 2 - x):min(brush_size_x, data.shape[0] - x + brush_size_x // 2),
            max(0, brush_size_y // 2 - y):min(brush_size_y, data.shape[1] - y + brush_size_y // 2)
        ]
    )
    return data


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
    # Find the idx from the POV of the brush
    idx_brush_window = list(product(brush_x_range, brush_y_range))
    # Only keep the idx where the brush is True
    mask_brush = (brush[*list(zip(*idx_brush_window))] == True)
    # Find the idx from the POV of the data
    idx_data_window = list(product(list(x_range), list(y_range)))
    # Use compress to filter the list of idx from the data POV
    idxs_conv = list(compress(idx_data_window,mask_brush))

    # return from_list_2ids_to_2list_id(idxs_conv)
    return idxs_conv
