import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import signal as sig

import ceviche_challenges
import brushes


NANOMETERS = ceviche_challenges.units.nm

spec = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_2umx2um_spec(
    wg_width=400 * NANOMETERS,
    variable_region_size=(1600 * NANOMETERS, 1600 * NANOMETERS),
    cladding_permittivity=2.25
)
params = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_sim_params(
    resolution=10 * NANOMETERS,
)
model = ceviche_challenges.waveguide_bend.model.WaveguideBendModel(params, spec)

design = np.ones(model.design_variable_shape)
# design = np.random.rand(*model.design_variable_shape)
brush = brushes.notched_square_brush(10, 1)
print(model.__dict__)
# print(model.compute_cost(design))
# exit()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

updated_density = model.density(design)
ax[0].imshow(updated_density.T, origin='lower', interpolation='none')

eroded = ndi.grey_erosion(updated_density, structure=brush)
# eroded = ndi.binary_erosion(design, structure=brush)
dilated = ndi.grey_dilation(eroded, structure=brush)
ax[1].imshow(dilated.T, origin='lower', interpolation='none')

dilated2 = ndi.grey_dilation(updated_density, structure=brush)
eroded2 = ndi.grey_erosion(dilated2, structure=brush)
ax[2].imshow(eroded2.T, origin='lower', interpolation='none')

plt.savefig('draft')
