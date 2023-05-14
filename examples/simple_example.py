import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from gruyere.brushes import notched_square_brush, show_mask
from gruyere.design import _initialize_design
from gruyere.touches import add_solid_touches


np.random.seed(42)


size = 10
my_brush = notched_square_brush(5, 1) * 1.0
# touch = [2, 2]
touches = [[2, 2], [3, 3]]

my_brush_convolved = convolve2d(my_brush, my_brush)

design_blank = _initialize_design(np.zeros((size, size)))

expected_design = np.zeros_like(design_blank.x)
list_idxs_touches = tuple(zip(*touches))
expected_design[list_idxs_touches] = 1.0

fig, ax = plt.subplots(1, 4)
ax[0].imshow(expected_design, origin='lower')
ax[1].imshow(my_brush != 0.0, origin='lower')
ax[2].imshow(my_brush_convolved != 0.0, origin='lower')

# exit()

final_design = add_solid_touches(
    design_blank,
    idx_touches=touches,
    brush=my_brush,
    brush_convolved=my_brush_convolved
)

ax[3].imshow(final_design.x, origin='lower')
plt.savefig('simple_example_results.png')
