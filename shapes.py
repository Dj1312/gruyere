# import numpy as np
# from scipy import ndimage


# # Snippet from mfschubert
# # https://github.com/mfschubert/topology/blob/bcd8f66c8c25ff4aac4d9df96730834b7641812b/shapes.py
# PLUS_KERNEL = onp.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


# def circle(diameter: int, padding: int) -> np.ndarray:
#     """Creates a pixelated circle with the given diameter."""
#     _validate_int(diameter, padding)
#     if diameter < 1:
#         raise ValueError(f"`diameter` must be positive, but got {diameter}.")
#     d = onp.arange(-diameter / 2 + 0.5, diameter / 2)
#     distance_squared = d[:, onp.newaxis] ** 2 + d[onp.newaxis, :] ** 2
#     kernel = distance_squared < (diameter / 2) ** 2
#     if diameter > 2:
#         # By convention we require that if the diameter is greater than `2`, 
#         # the kernel must be realizable with the plus-shaped kernel.
#         kernel = ndimage.binary_opening(kernel, PLUS_KERNEL)
#     return symmetric_pad(kernel, padding)


# def symmetric_pad(x: onp.ndarray, padding: int) -> onp.ndarray:
#     """Symmetrically pads `x` by the specified amount."""
#     return onp.pad(x, ((padding, padding), (padding, padding)))


# def _rotate(x: onp.ndarray, y: onp.ndarray, angle: onp.ndarray) -> onp.ndarray:
#     """Rotates `(x, y)` by the specified angle."""
#     magnitude = onp.sqrt(x**2 + y**2)
#     xy_angle = onp.angle(x + 1j * y)
#     rot = magnitude * onp.exp(1j * (xy_angle + angle))
#     return rot.real, rot.imag


# def trim_zeros(x: onp.ndarray) -> onp.ndarray:
#     """Trims the nonzero elements from `x`."""
#     i, j = onp.nonzero(x)
#     return x[onp.amin(i) : onp.amax(i) + 1, onp.amin(j) : onp.amax(j) + 1]


# def _validate_int(*args):
#     """Validates that all arguments are integers."""
#     if any([not isinstance(x, int) for x in args]):
#         raise ValueError(f"Expected ints but got types {[x.type for x in args]}")
