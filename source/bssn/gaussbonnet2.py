import numpy as np

from core.grid import Grid
from bssn.tensoralgebra import *
from bssn.bssnvars import *

# Constants for tensor algebra
one_sixth = 1.0/6.0
one_third = 1.0/3.0
two_thirds = 2.0/3.0
four_thirds = 4.0/3.0
two_nine = 2.0/9.0
third_two = 3.0/2.0

def compute_L_GB(bssn_vars, bssn_rhs, d1, d2, matter, grid, background):
    """
    Compute the Gauss-Bonnet term L_GB = f'(u) * G for the scalar field equation.
    Assumes f(u) = alpha * u (linear coupling), where u is the scalar field.
    Returns L_GB as a 1D array of shape (N,) to be added to dvdt in scalarmatter.py.
    
    Parameters:
    - bssn_vars: BSSNVars object containing BSSN variables.
    - d1: BSSNFirstDerivs object with first derivatives.
    - d2: BSSNSecondDerivs object with second derivatives.
    - matter: ScalarMatter object containing u, v, and their derivatives.
    - grid: Grid object with radial coordinate r and other grid properties.
    - background: SphericalBackground object with background metric data.
    """
    r = grid.r
    N = grid.num_points

    # Extract BSSN variables
    K = bssn_vars.K  # Trace of extrinsic curvature
    phi = bssn_vars.phi  # Conformal factor
    chi = np.exp(-4.0 * phi)  # chi = e^(-4*phi)
    chii = 1.0 / chi  # Inverse of chi
    lapse = bssn_vars.lapse  # Lapse function
    shift_U = bssn_vars.shift_U

    # Extract scalar field variables from matter
    u = matter.u  # Scalar field
    d1_u = matter.d1_u  # First derivative of u
    emtensor  = matter.get_emtensor(r, bssn_vars, background)

    # Metric and extrinsic curvature tensors
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
    gamma_LL = chi * bar_gamma_LL  # Physical metric gamma_ij = chi * bar_gamma_ij
    gamma_UU = chii * bar_gamma_UU  # Inverse physical metric
    bar_A_LL = get_bar_A_LL(r, bssn_vars, background)  
    bar_A_UU = get_bar_A_UU(r, bssn_vars, background) 

    # Derivatives
    d1_phi = d1.phi  # Partial_i phi
    d2_phi = d2.phi  # Partial_i partial_j phi
    d1_K = d1.K  # Partial_i K
    d1_lapse = d1.lapse  # Partial_i alpha
    d2_lapse = d2.lapse  # Partial_i partial_j alpha
    d1_A_LL = d1.a_LL  # Partial_k A_ij

    # Compute connections
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    # Recompute M_ij (Eq. 5) 
    #Here i assumed that in LLiberts paper bar_ricci = ricci (in our case for spherical symmetry)
    bar_Ricci = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL, bssn_vars.lambda_U, d1.lambda_U,
                                 Delta_U, Delta_ULL, Delta_LLL, bar_gamma_UU, bar_gamma_LL, background)
    AikAkj = get_AikAkj(bar_A_LL, bar_gamma_UU)
    M_LL = (bar_Ricci 
            + chii * two_nine * bar_gamma_LL * K[:, np.newaxis, np.newaxis] * K[:, np.newaxis, np.newaxis]
            + chii * one_third * K[:, np.newaxis, np.newaxis] * bar_A_LL
            - AikAkj)
    
    trace_M = get_trace(M_LL, gamma_UU)

    # Recompute N_i (Eq. 6) 
    bar_div_A_L = np.einsum('xjik->xi', d1_A_LL) - np.einsum('xkjl,xlk->xj', bar_chris, bar_A_LL)
    d1_chi = -4.0 * chi[:, np.newaxis] * d1_phi  # Partial_j chi
    N_L = (bar_div_A_L
           - third_two* chii[:, np.newaxis]*d1_chi
           - two_thirds* d1_K)

    #Compute L_GB 
    get_bssn_rhs(bssn_rhs, r, bssn_vars, d1, d2, background, emtensor)
    ##################################################
    #LINE1
    #Compute laplacian of lapse first
    laplacian_alpha = (np.einsum('xij,xij->x', bar_gamma_UU, d2_alpha)
                        - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1_alpha))
    
    #acces dKdt
    dKdt = bssn_rhs.K
    ilapse = 1/lapse

    line1_GB = -four_thirds*trace_M*(ilapse *dKdt
                                     + ilapse * laplacian_alpha
                                     - AikAkj
                                     - one_third * K * K)
    ##################################################
    #LINE2

    #Raising the indices
    M_UU = np.einsum('xik,xjl,xij->xkl', gamma_UU, gamma_UU, M_LL)
    #We have to add axis 2 and three to trace_M so numpy can multiply it with the gamma_UU
    TraceFree_M_UU = M_UU - one_third*(trace_M[:, np.newaxis, np.newaxis] * gamma_UU)

    dAdt = bssn_rhs.a_LL

    Shift_U = background.inverse_scaling_vector * bssn_vars.shift_U
    d1_Shift_U = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,np.newaxis] 
                     + d1.shift_U * background.inverse_scaling_vector[:,:,np.newaxis])



    line2_GB = 8*TraceFree_M_UU*(ilapse * dAdt
                                 + ilapse*(np.einsum('xjk,xjl->xkl', bar_A_LL, d1_Shift_U)
                                                +np.einsum('xjl,xjk->xkl', bar_A_LL, d1_Shift_U))
                                 + ilapse )
        
    







  

    return L_GB