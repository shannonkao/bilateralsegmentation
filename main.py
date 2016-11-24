from scipy.misc import imread, imsave
from matplotlib import *
import numpy as np
import os

from util import rgb2yuv, yuv2rgb
from BilateralGrid import BilateralGrid
from BilateralSolver import BilateralSolver

data_folder = os.path.abspath(os.path.join(os.path.curdir, 'data', 'depth_superres'))
output_folder = os.path.abspath(os.path.join(os.path.curdir, 'output'))

# The RGB image that whose edges we will respect
reference = imread(os.path.join(data_folder, 'reference.png'))
# The 1D image whose values we would like to filter
target = imread(os.path.join(data_folder, 'target.png'))
# A confidence image, representing how much we trust the values in "target".
# Pixels with zero confidence are ignored.
# Confidence can be set to all (2^16-1)'s to effectively disable it.
confidence = imread(os.path.join(data_folder, 'confidence.png'))

im_shape = reference.shape[:2]
assert(im_shape[0] == target.shape[0])
assert(im_shape[1] == target.shape[1])
assert(im_shape[0] == confidence.shape[0])
assert(im_shape[1] == confidence.shape[1])

# figure(figsize=(14, 20))
# subplot(311)
# imshow(reference)
# title('reference')
# subplot(312)
# imshow(confidence)
# title('confidence')
# subplot(313)
# imshow(target)
# title('target')

grid_params = {
    'sigma_luma' : 4, # Brightness bandwidth
    'sigma_chroma': 4, # Color bandwidth
    'sigma_spatial': 8 # Spatial bandwidth
}

bs_params = {
    'lam': 128, # The strength of the smoothness parameter
    'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
    'cg_tol': 1e-5, # The tolerance on the convergence in PCG
    'cg_maxiter': 25 # The number of PCG iterations
}

grid = BilateralGrid(reference, **grid_params)

t = target.reshape(-1, 1).astype(np.double) / (pow(2,16)-1)
c = confidence.reshape(-1, 1).astype(np.double) / (pow(2,16)-1)
tc_filt = grid.filter(t * c)
c_filt = grid.filter(c)
output_filter = (tc_filt / c_filt).reshape(im_shape)

output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(im_shape)

imargs = dict(vmin=0, vmax=1)
# figure(figsize=(14, 24))
# subplot(311)
# imshow(t.reshape(im_shape), **imargs)
# title('input')
# subplot(312)
# imshow(output_filter, **imargs)
# title('bilateral filter')
# subplot(313)
# imshow(output_solver, **imargs)
# title('bilateral solver')

imsave(os.path.join(output_folder, 'output.png'), output_solver)