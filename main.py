from scipy.misc import imread, imsave
from matplotlib import *
import numpy as np
import os

from BilateralGrid import BilateralGrid
from BilateralSolver import BilateralSolver

data_folder = os.path.abspath(os.path.join(os.path.curdir, 'data', 'depth_superres'))
output_folder = os.path.abspath(os.path.join(os.path.curdir, 'output'))

def smoothImage(ref_str, target_str, conf_str):
    # The RGB image that whose edges we will respect
    reference = imread(os.path.join(data_folder, ref_str))
    # The 1D image whose values we would like to filter
    target = imread(os.path.join(data_folder, target_str))
    # A confidence image, representing how much we trust the values in "target".
    # Pixels with zero confidence are ignored.
    # Confidence can be set to all (2^16-1)'s to effectively disable it.
    confidence = imread(os.path.join(data_folder, conf_str))

    im_shape = reference.shape[:2]
    assert(im_shape[0] == target.shape[0])
    assert(im_shape[1] == target.shape[1])
    assert(im_shape[0] == confidence.shape[0])
    assert(im_shape[1] == confidence.shape[1])

    # Set parameters
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

    # Construct hard bilateral grid
    grid = BilateralGrid(reference, **grid_params)

    # Apply bilateral solver
    t = target.reshape(-1, 1).astype(np.double) / (pow(2,16)-1)
    c = confidence.reshape(-1, 1).astype(np.double) / (pow(2,16)-1)
    output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(im_shape)

    # Save the output
    imsave(os.path.join(output_folder, 'output.png'), output_solver)

smoothImage('reference.png', 'target.png', 'confidence.png')