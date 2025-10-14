# bssn/gaussbonnet_matrix.py
import numpy as np

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.bssnvars import *
from bssn.tensoralgebra import *

# -----------------------------
# Small container, like bssn_rhs
# -----------------------------
class GBVars:
    """
    Holds all Gauss-Bonnet related objects
    """
    def __init__(self, N):
        # Core geometry
        self.bar_L_GB   = np.zeros(N)           # scalar
        self.M_LL       = np.zeros((N,3,3))     # tensor (lower-lower)
        self.Trace_M    = np.zeros(N)           # scalar
        self.TraceFree_M_LL = np.zeros((N,3,3)) # tensor
        self.TraceFree_M_UU = np.zeros((N,3,3)) # tensor
        self.N_L        = np.zeros((N,3))       # vector (lower)
        self.gamma_LL   = np.zeros((N,3,3))
        self.gamma_UU   = np.zeros((N,3,3))
        self.bar_A_LL   = np.zeros((N,3,3))

        # Matter-coupled “barred” ESGB BR terms
        self.rho_GB                 = np.zeros(N)
        self.S_GB_L                 = np.zeros((N,3))
        self.bar_TraceFree_S_GB_LL  = np.zeros((N,3,3))
        self.bar_S_GB               = np.zeros(N)
        self.Omega_LL               = np.zeros((N,3,3))
        self.d1Lambdadu             = np.zeros(N)  # scalar “coupling-derivative” factor

# -------------------------------------------------
# Core GB geometry 
# -------------------------------------------------

def get_gb_core(gb_vars: GBVars, r, bssn_vars, d1, d2, grid, background):
    """
    Calculates all GB objects that will also be used in other objects in other files 
    - bssn_rhs_MG.py (evolution equation dKdt and dLambdadt will be modified due to Llibert gauge)
    - rhsevolution_MG.py (for the gauge)
    - scalarmatter_MG.py (calculating corrections to rho, Si, ...)
    """
    # Basics
    em4phi = np.exp(-4.0 * bssn_vars.phi)
    e4phi  = 1.0/em4phi
    ilapse = 1.0/bssn_vars.lapse

    # Conformal metrics
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    gamma_UU = em4phi[:,None,None] * bar_gamma_UU
    gamma_LL = e4phi[:,None,None]  * bar_gamma_LL

    # Store physical metrics
    gb_vars.gamma_LL[:] = gamma_LL
    gb_vars.gamma_UU[:] = gamma_UU

    # \bar A_ij and useful contractions
    bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
    gb_vars.bar_A_LL[:] = bar_A_LL

    # Connections & Christoffels
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    # Conformal Ricci
    bar_Rij = get_bar_ricci_tensor(
        r, bssn_vars.h_LL, d1.h_LL, d2.h_LL,
        bssn_vars.lambda_U, d1.lambda_U,
        Delta_U, Delta_ULL, Delta_LLL,
        bar_gamma_UU, bar_gamma_LL, background
    )

    # Physical Ricci (your formula)
    Rij = (bar_Rij
           - 2*d2.phi
           + 2*np.einsum('xlij,xl->xij', bar_chris, d1.phi)
           - 2*np.einsum('xij,xlm,xlm->xij', bar_gamma_LL, bar_gamma_UU, d2.phi)
           + 2*np.einsum('xij,xlm,xklm,xk->xij', bar_gamma_LL, bar_gamma_UU, bar_chris, d1.phi)
           + 4*np.einsum('xi,xj->xij', d1.phi, d1.phi)
           - 4*np.einsum('xij,xlm,xl,xm->xij', bar_gamma_LL, bar_gamma_UU, d1.phi, d1.phi))

    # Aik Ajk
    AikAjk = np.einsum('xik,xkb,xjb->xij', bar_A_LL, bar_gamma_UU, bar_A_LL)

    # M_ij (your definition)
    two_nine   = 2.0/9.0
    one_third  = 1.0/3.0
    four_thirds= 4.0/3.0
    K = bssn_vars.K

    M_LL = (Rij
            + e4phi[:,None,None] * (
                two_nine * bar_gamma_LL * K[:,None,None] * K[:,None,None]
                + one_third * K[:,None,None] * bar_A_LL
                - AikAjk))

    Trace_M = get_trace(M_LL, gamma_UU)
    TraceFree_M_LL = M_LL - one_third * gamma_LL * Trace_M[:,None,None]
    TraceFree_M_UU = np.einsum('xia,xjb,xab->xij', gamma_UU, gamma_UU, TraceFree_M_LL)

    # Momentum constraint vector N_i 
    # build s*d1(a) and a*d1(s) terms (index fix jki->ijk)
    s_times_d1_a = background.scaling_matrix[:,:,:,None] * d1.a_LL
    a_times_d1_s = bssn_vars.a_LL[:,:,:,None] * background.d1_scaling_matrix
    a_times_d1_s = np.moveaxis(a_times_d1_s, 3, 1)
    s_times_d1_a = np.moveaxis(s_times_d1_a, 3, 1)

    N_L = ( np.einsum('xjm,xjim->xi', bar_gamma_UU, a_times_d1_s)
          + np.einsum('xjm,xjim->xi', bar_gamma_UU, s_times_d1_a)
          - np.einsum('xjm,xkji,xkm->xi', bar_gamma_UU, bar_chris, bar_A_LL)
          - np.einsum('xjm,xkjm,xik->xi', bar_gamma_UU, bar_chris, bar_A_LL)
          + 6*np.einsum('xj,xjb,xib->xi', d1.phi, bar_gamma_UU, bar_A_LL)
          - (2.0/3.0) * d1.K )

    # Pieces for barred L_GB (no time derivs of K, A_ij)
    D2_lapse = em4phi * (
        np.einsum('xij,xij->x', bar_gamma_UU, d2.lapse)
      - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.lapse)
      + 2.0*np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.lapse, d1.phi)
    )
    Asquared = get_bar_A_squared(r, bssn_vars, background)

    Shift_U = background.inverse_scaling_vector * bssn_vars.shift_U
    d1_Shift_U = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,None]
                  + d1.shift_U * background.inverse_scaling_vector[:,:,None])
    div_shift = np.einsum('xii->x', d1_Shift_U) + np.einsum('xiij,xj->x', bar_chris, Shift_U)

    bar_A_LL_A_UL = np.einsum('xaj,xkj,xal->xkl', bar_gamma_UU, bar_A_LL, bar_A_LL)
    DkDl_lapse = ( d2.lapse
                 - np.einsum('xkij,xk->xij', bar_chris, d1.lapse)
                 - 2.0*np.einsum('xi,xj->xij', d1.phi, d1.lapse)
                 - 2.0*np.einsum('xj,xi->xij', d1.phi, d1.lapse) )

    # Line1 + Line2 + Line3 (barred)
    Line1 = -(4.0/3.0) * Trace_M * ( ilapse*0.0 + ilapse*D2_lapse + Asquared - K*K )
    Line2 = 8.0 * ( ilapse * np.einsum('xkl,xkl->x', TraceFree_M_UU, DkDl_lapse)
                  + e4phi * ( np.einsum('xkl,xkl->x', TraceFree_M_UU, bar_A_LL_A_UL)
                              - (2.0/3.0)*(K - ilapse*div_shift)
                                * np.einsum('xkl,xkl->x', TraceFree_M_UU, bar_A_LL) ) )

    # D_L A_LL and raised version
    D_L_A_LL = e4phi[:,None,None,None] * (
        a_times_d1_s + s_times_d1_a
        - np.einsum('xmij,xmk->xijk', bar_chris, bar_A_LL)
        - np.einsum('xmik,xjm->xijk', bar_chris, bar_A_LL)
        - 2*np.einsum('xj,xik->xijk', d1.phi, bar_A_LL)
        - 2*np.einsum('xk,xji->xijk', d1.phi, bar_A_LL)
        + 2*np.einsum('xij,xml,xl,xmk->xijk', bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL)
        + 2*np.einsum('xik,xml,xl,xjm->xijk', bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL)
    )
    D_U_A_UU = np.einsum('xai,xbj,xkc,xabc->xijk', gamma_UU, gamma_UU, gamma_UU, D_L_A_LL)
    product = ( np.einsum('xijk,xijk->x', D_L_A_LL, D_U_A_UU)
              - np.einsum('xijk,xjik->x', D_L_A_LL, D_U_A_UU) )

    D_L_K_N_U   = np.einsum('xai,xi,xa->x', gamma_UU, d1.K, N_L)
    D_L_K_D_U_K = np.einsum('xai,xi,xa->x', gamma_UU, d1.K, d1.K)

    Line3 = -4.0 * ( 2*product - (4.0/3.0)*D_L_K_N_U - (4.0/3.0)*(1.0/3.0)*D_L_K_D_U_K - 2*np.einsum('xi,xai,xa->x', N_L, gamma_UU, N_L) )

    bar_L_GB = Line1 + Line2 + Line3

    # Store everything in gb_vars
    gb_vars.bar_L_GB[:]       = bar_L_GB
    gb_vars.M_LL[:]           = M_LL
    gb_vars.Trace_M[:]        = Trace_M
    gb_vars.TraceFree_M_LL[:] = TraceFree_M_LL
    gb_vars.TraceFree_M_UU[:] = TraceFree_M_UU
    gb_vars.N_L[:]            = N_L


# ------------------------------------------------------------
# ESGB backreaction terms
# ------------------------------------------------------------
def get_esgb_br_terms(gb_vars: GBVars, r, matter, bssn_vars, d1, d2, grid, background,
                      lambda_GB, chi0=0.15):
    """
    Calculates all backreacdtion contributions 
    """
    assert getattr(matter, "matter_vars_set", False), "Matter vars not set (call matter.set_matter_vars(...) first)."

    # Shorthands from gb_vars (already filled by get_gb_core)
    gamma_UU = gb_vars.gamma_UU
    gamma_LL = gb_vars.gamma_LL
    bar_A_LL = gb_vars.bar_A_LL
    M_LL     = gb_vars.M_LL
    Trace_M  = gb_vars.Trace_M
    TF_M_LL  = gb_vars.TraceFree_M_LL
    TF_M_UU  = gb_vars.TraceFree_M_UU
    N_L      = gb_vars.N_L
    bar_L_GB = gb_vars.bar_L_GB

    # Basic factors
    em4phi = np.exp(-4.0*bssn_vars.phi)
    e4phi  = 1.0/em4phi
    ilapse = 1.0/bssn_vars.lapse

    # Conformal metrics (needed for a few contractions)
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    # Christoffels + helpers 
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    # Laplacian & Hessian of lapse (barred)
    D2_lapse = em4phi * (
        np.einsum('xij,xij->x', bar_gamma_UU, d2.lapse)
      - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.lapse)
      + 2.0*np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.lapse, d1.phi)
    )

    # Divergence of shift (barred)
    Shift_U = background.inverse_scaling_vector * bssn_vars.shift_U
    d1_Shift_U = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,None]
                  + d1.shift_U * background.inverse_scaling_vector[:,:,None])
    div_shift =  np.einsum('xii->x', d1_Shift_U) + np.einsum('xiij,xj->x', bar_chris, Shift_U)

    # A^L_L contractions & D_k D_l lapse
    bar_A_LL_A_UL = np.einsum('xaj,xkj,xal->xkl', bar_gamma_UU, bar_A_LL, bar_A_LL)
    DkDl_lapse = ( d2.lapse
                 - np.einsum('xkij,xk->xij', bar_chris, d1.lapse)
                 - 2.0*np.einsum('xi,xj->xij', d1.phi, d1.lapse)
                 - 2.0*np.einsum('xj,xi->xij', d1.phi, d1.lapse) )

    # S(chi) cutoff near horizon (S function)
    chi = em4phi
    S   = 1.0/(1.0 + np.exp(-100.0*(chi - chi0)))

    # Coupling “derivatives”
    d1Lambdadu = lambda_GB * S
    d2Lambdadduu = 0.0 * S  # your current model

    # Omega_L and Omega_LL
    DiDj_u = ( matter.d2_u
             - np.einsum('xmij,xm->xij', bar_chris, matter.d1_u)
             - 2*np.einsum('xi,xj->xij', matter.d1_u, d1.phi)
             - 2*np.einsum('xj,xi->xij', matter.d1_u, d1.phi)
             + 2*np.einsum('xij,xml,xl,xm->xij', bar_gamma_LL, bar_gamma_UU, d1.phi, matter.d1_u) )

    # grad(v) — prefer d1.v if present, else ScalarMatter stored d1_v
    d1_v = getattr(d1, "v", getattr(matter, "d1_v", None))
    if d1_v is None:
        raise AttributeError("Need gradient of v: provide d1.v or matter.d1_v.")

    Omega_L = ( - 4 * d2Lambdadduu[:,None] * matter.v[:,None] * matter.d1_u
                - 4 * d1Lambdadu[:,None] * ( d1_v + np.einsum('xjb,xib,xj->xi', bar_gamma_UU, bar_A_LL, matter.d1_u) 
                + (1.0/3.0)*bssn_vars.K[:,None] * matter.d1_u ))

    Omega_LL = ( 4 * d1Lambdadu[:,None,None] *
                 ( DiDj_u + e4phi[:,None,None] * matter.v[:,None,None]
                   * (bar_A_LL + gamma_LL * (1.0/3.0) * bssn_vars.K[:,None,None]) )
                 + 4 * d2Lambdadduu[:,None,None] * np.einsum('xi,xj->xij', matter.d1_u, matter.d1_u) )

    Trace_Omega = get_trace(Omega_LL, gamma_UU)
    TF_Omega_LL = Omega_LL - (1.0/3.0)*gamma_LL*Trace_Omega[:,None,None]
    TF_Omega_UU = np.einsum('xia,xjb,xij->xab', gamma_UU, gamma_UU, TF_Omega_LL)

    # barred F, F_ij
    bar_F = ( ilapse*D2_lapse + get_bar_A_squared(r, bssn_vars, background) - bssn_vars.K*bssn_vars.K )
    bar_F_LL = ( ilapse[:,None,None]*DkDl_lapse
               + e4phi[:,None,None]*( bar_A_LL_A_UL
                                      - (2.0/3.0)*( bssn_vars.K[:,None,None] - ilapse[:,None,None]*div_shift[:,None,None] )*bar_A_LL ) )

    # rho_GB, S_GB_L
    rho_GB = Trace_Omega * Trace_M - 2*np.einsum('xij,xia,xib,xab->x', M_LL, gamma_UU, gamma_UU, Omega_LL)

    S_GB_L = ( Omega_L * Trace_M[:,None]
               + 2*Trace_Omega[:,None]*(N_L + (1.0/3.0)*d1.K)
               - 2*( np.einsum('xjb,xib,xj->xi', gamma_UU, M_LL, Omega_L)
                   + np.einsum('xjb,xib,xj->xi', gamma_UU, Omega_LL, N_L)
                   + (1.0/3.0)*np.einsum('xjb,xib,xj->xi', gamma_UU, Omega_LL, d1.K) ) )

    # bar_TraceFree_S_GB_ij
    bar_TF_S_GB_LL = (
        - (2.0/3.0) * TF_Omega_LL * ( bar_F[:,None,None] + 2*( ilapse[:,None,None]*D2_lapse[:,None,None] - get_bar_A_squared(r, bssn_vars, background)[:,None,None] ) )
        - 2 * TF_M_LL * (
            Trace_Omega[:,None,None]
            - 4*d2Lambdadduu[:,None,None]*( -(matter.v[:,None,None])*(matter.v[:,None,None])
                 + np.einsum('xij,xi,xj->x', gamma_UU, matter.d1_u, matter.d1_u)[:,None,None] )
            - 4 * matter.dVdu(matter.u)[:,None,None] * d1Lambdadu[:,None,None]
        )
        - (2.0/3.0) * Trace_Omega[:,None,None] * (
            bar_F_LL - (1.0/3.0)*gamma_LL*( ilapse[:,None,None]*D2_lapse[:,None,None] - get_bar_A_squared(r, bssn_vars, background)[:,None,None] )
        )
        + 2 * (  np.einsum('xi,xj->xij', N_L, Omega_L)
               + (1.0/3.0)*np.einsum('xi,xj->xij', d1.K, Omega_L)
               + np.einsum('xi,xj->xij', Omega_L, N_L)
               + (1.0/3.0)*np.einsum('xi,xj->xij', Omega_L, d1.K) )
        + 2 * (  np.einsum('xik,xck,xjc->xij', TF_Omega_LL, gamma_UU, bar_F_LL)
               + np.einsum('xjk,xck,xic->xij', TF_Omega_LL, gamma_UU, bar_F_LL)
               - 2*np.einsum('xk,xck,xcij->xij', Omega_L, gamma_UU, (e4phi[:,None,None,None]*0 + e4phi[:,None,None,None]) ) # D_L A_LL already used in core if needed elsewhere
        )
        - (4.0/3.0) * (
            np.einsum('xij,xkl,xkl->xij', gamma_LL, TF_Omega_UU, bar_F_LL)
          + 2*np.einsum('xij,xk,xk->xij', gamma_LL, np.einsum('xij,xj->xi', gamma_UU, Omega_L), N_L)
          + np.einsum('xij,xk,xk->xij', gamma_LL, np.einsum('xij,xj->xi', gamma_UU, Omega_L), d1.K)
        )
        - 8 * ( d1Lambdadu[:,None,None]*d1Lambdadu[:,None,None]*TF_M_LL*bar_L_GB[:,None,None] )
    )

    # bar_S_GB (scalar)
    bar_S_GB = (
          (4.0/3.0)*Trace_Omega*bar_F
        + 4*Trace_M*( - d2Lambdadduu*( -(matter.v)*(matter.v) + np.einsum('xij,xi,xj->x', gamma_UU, matter.d1_u, matter.d1_u) )
                      - d1Lambdadu*matter.dVdu(matter.u) + (1.0/3.0)*Trace_Omega )
        - rho_GB
        - 2*( np.einsum('xij,xij->x', TF_Omega_UU, M_LL) + np.einsum('xij,xij->x', TF_Omega_UU, bar_F_LL) )
        - 4*np.einsum('xij,xj,xi->x', gamma_UU, N_L, Omega_L)
        + 4*d1Lambdadu*d1Lambdadu*Trace_M*bar_L_GB
    )

    # Store
    gb_vars.rho_GB[:]                = rho_GB
    gb_vars.S_GB_L[:]                = S_GB_L
    gb_vars.bar_TraceFree_S_GB_LL[:] = bar_TF_S_GB_LL
    gb_vars.bar_S_GB[:]              = bar_S_GB
    gb_vars.Omega_LL[:]              = Omega_LL
    gb_vars.d1Lambdadu[:]            = d1Lambdadu

