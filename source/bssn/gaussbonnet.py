import numpy as np

from core.grid import Grid
from bssn.tensoralgebra import *

def compute_L_GB(bssn_vars, d1, d2, rhs_dict, grid, background):
    r = grid.r
    N = grid.num_points

    # Extract variables and quantities
    K = bssn_vars.K                        # trace of extrinsic curvature
    A_LL = get_bar_A_LL(r, bssn_vars, background)  # physical A_ij
    A_UU = get_bar_A_UU(r, bssn_vars, background)  # A^ij
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    # Derivatives
    dK = d1.K                              # D_i K
    d1_alpha = d1.lapse                    # D_i alpha
    d2_alpha = d2.lapse                    # D_i D_j alpha
    d1_A_LL = d1.a_LL                      # D_k A_ij
    d2_A_LL = d2.a_LL                      # D_k D_l A_ij (if needed)

    # Lapse and conformal factor
    alpha = bssn_vars.lapse
    phi = bssn_vars.phi
    chi = np.exp(-4.0 * phi)

    # Get bar_R_ij (Ricci tensor)
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_Rij = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL, bssn_vars.lambda_U, d1.lambda_U,
                                   Delta_U, Delta_ULL, Delta_LLL, bar_gamma_UU, bar_gamma_LL, background)

    # Construction M_ij from Eq. (5) 



    # Construction N_i from Eq. (6)




    # Construction L^GB using expression in Eq. (4)
    L_GB = np.zeros(N)





    return L_GB