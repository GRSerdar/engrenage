import numpy as np

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.bssnvars import *
from bssn.tensoralgebra import *
from bssn.gaussbonnet_working import * #compute_L_GB


class ScalarMatter :
    """Represents the matter that sources the Einstein equation."""

    def __init__(self, a_scalar_mu=1.0) :
        self.scalar_mu = a_scalar_mu # this is an inverse length scale related to the scalar compton wavelength
        
        # Details for the matter state variables
        self.NUM_MATTER_VARS = 2
        self.VARIABLE_NAMES = ["u", "v"]
        self.PARITY = np.array([1, 1])
        self.ASYMP_POWER = np.array([0, 0])
        self.ASYMP_OFFSET = np.array([0, 0])
        self.idx_u = NUM_BSSN_VARS
        self.idx_v = NUM_BSSN_VARS + 1
        self.indices = np.array([self.idx_u, self.idx_v])
        self.matter_vars_set = False
        self.u = []
        self.v = []
        self.d1_u = []
        self.d2_u = []
        self.d1_v = []
        self.advec_u = []
        self.advec_v = []
        
    # The scalar potential
    def V_of_u(self, u) :
        return 0.5 * self.scalar_mu * self.scalar_mu * u * u

    # Derivative of scalar potential
    def dVdu(self, u) :
        return self.scalar_mu * self.scalar_mu * u
    
    ###########################################################################################################################
    # The coupling function of the gauss bonnet term can change (now it is lambda(phi) = lambda_GB phi)
    
    ###########################################################################################################################

    # def get_emtensor(self, r, bssn_vars, background) :
    def get_Extra_BR_terms(self, r, bssn_vars, bssn_d1, bssn_d2, bssn_rhs, grid, background):

        assert self.matter_vars_set, 'Matter vars not set'
        
        N = np.size(r) 
        scalar_emtensor = EMTensor(N)
        em4phi = np.exp(-4.0 * bssn_vars.phi)
        e4phi = 1/em4phi

        # Barred Metric and extrinsic curvature tensors
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

        # Physical Metrics (since we will need a few times in this file)
        gamma_UU = em4phi[:,np.newaxis,np.newaxis] *bar_gamma_UU
        gamma_LL = e4phi[:,np.newaxis,np.newaxis] * bar_gamma_LL

        bar_A_LL = get_bar_A_LL(r, bssn_vars, background) 
        
        M_LL = compute_L_GB(bssn_vars, bssn_rhs, bssn_d1, bssn_d2, grid, background)[1] #idx=1 corresponds with M_LL
        N_L = compute_L_GB(bssn_vars, bssn_rhs, bssn_d1, bssn_d2, grid, background)[2] #idx=2 corresponds with N_L
        Trace_M = get_trace(M_LL, gamma_UU)
        
        # The connections + Christoffel symbols
        Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, bssn_vars.h_LL, bssn_d1.h_LL, background)
        bar_chris = get_bar_christoffel(r, Delta_ULL, background)

        chi = np.exp(-4.0* bssn_vars.phi) 
        chi0 = 0.15
        function_of_lambda= (1)/(1+np.exp(-100*(chi-chi0)))
    
        def d1_Lambda_d1_u(lambda_GB):
            return (lambda_GB*function_of_lambda)
    
        def d2_Lambda_d2_u(lambda_GB):
            return 0*function_of_lambda

        ###########################################################################################################################
        ###########################################################################################################################
        # Define extra objects to compute the modified rho, S_i and S_ij

        # Currently just setting a local value for lambda_GB (idea is to be able to loop over some values in notebook)
        lambda_GB =  0.05

        # \Omega_i (Note that the first term can be zero if coupling funciton is only of first order)
        Omega_L = (- 4 * d2_Lambda_d2_u(lambda_GB)[:,np.newaxis] * self.v[:,np.newaxis]* self.d1_u
                   - 4 * d1_Lambda_d1_u(lambda_GB)[:,np.newaxis] * (self.d1_v + np.einsum("xjb, xib,xj->xi",bar_gamma_UU, bar_A_LL,self.d1_u))
                   + one_third * bssn_vars.K[:,np.newaxis] * self.d1_u)
        
        # D_i D_j u (scalar field)
        DiDj_u = (self.d2_u 
                  - np.einsum("xmij, xm->xij",bar_chris, self.d1_u)
                  - 2 * np.einsum("xi, xj->xij",self.d1_u, bssn_d1.phi)
                  - 2 * np.einsum("xj, xi->xij",self.d1_u,bssn_d1.phi)
                  + 2 * np.einsum("xij, xml, xl, xm->xij",bar_gamma_LL, bar_gamma_UU, bssn_d1.phi, self.d1_u))
        
        # Omega_ij
        Omega_LL = ( 4 * d1_Lambda_d1_u(lambda_GB)[:,np.newaxis, np.newaxis] * (DiDj_u + e4phi[:,np.newaxis,np.newaxis] * self.v[:,np.newaxis,np.newaxis] * (bar_A_LL + bar_gamma_LL * one_third * bssn_vars.K[:,np.newaxis,np.newaxis]))
                    +4 * d2_Lambda_d2_u(lambda_GB)[:,np.newaxis, np.newaxis] * np.einsum('xi,xj->xij', self.d1_u, self.d1_u))
        
        # \Omega = gamma^ij Omega_ij
        Trace_Omega = get_trace(Omega_LL, gamma_UU)

        ##### Matter Term Corrections #########################################################################################
    
        # Gauss bonnet correction to rho
        rho_GB = Trace_Omega*Trace_M - 2*np.einsum("xij, xia, xib, xab->x",M_LL, gamma_UU, gamma_UU, Omega_LL)

        # Gauss bonnet correction to S_i 
        S_GB_L = (Omega_L * Trace_M[:,np.newaxis] 
                  + 2*Trace_Omega[:,np.newaxis]*(N_L + one_third * bssn_d1.K)
                  - 2*(np.einsum("xjb, xib, xj->xi",gamma_UU, M_LL, Omega_L)
                       + np.einsum("xjb,xib,xj->xi",gamma_UU, Omega_LL, N_L)
                       + one_third * np.einsum("xjb, xib, xj->xi",gamma_UU, Omega_LL, bssn_d1.K)))
        
        # Gauss bonnet correction to S_ij
        S_GB_LL = 0
        
        return (rho_GB, S_GB_L, S_GB_LL, Trace_M, N_L)

    def get_emtensor(self, r, bssn_vars, bssn_d1, bssn_d2, bssn_rhs, grid, background):
        
        assert self.matter_vars_set, 'Matter vars not set'
        
        N = np.size(r) 
        scalar_emtensor = EMTensor(N)
        em4phi = np.exp(-4.0 * bssn_vars.phi)
        e4phi = 1/em4phi

        # Barred Metric and extrinsic curvature tensors
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

        # Physical Metrics (since we will need a few times in this file)
        # gamma_UU = em4phi[:,np.newaxis,np.newaxis] *bar_gamma_UU

        ###########################################################################################################################
        ##### Matter Term Corrections ############################################################################################
        
        rho_GB, S_GB_L, S_GB_LL,Trace_M, N_L = self.get_Extra_BR_terms(r, bssn_vars, bssn_d1, bssn_d2, bssn_rhs, grid,  background)

        ###########################################################################################################################
        ###########################################################################################################################
        
        # The potential V(u) = 1/2 mu^2 u^2
        scalar_emtensor.rho = (  0.5 * self.v * self.v
                               + 0.5 * em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, self.d1_u, self.d1_u)
                               + self.V_of_u(self.u) ) + rho_GB
        
        scalar_emtensor.Si = - self.v[:,np.newaxis] * self.d1_u + S_GB_L
        
        # Useful quantity Vt
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        Vt = - self.v * self.v + em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, self.d1_u, self.d1_u)
        
        # Need to get the scalar factor in the right array dimension
        scalar_factor = - ((0.5 * Vt  + self.V_of_u(self.u)) / em4phi)
        scalar_emtensor.Sij = (scalar_factor[:,np.newaxis,np.newaxis] * bar_gamma_LL
                                   + np.einsum('xi,xj->xij', self.d1_u, self.d1_u)) + S_GB_LL
        
        # The trace of S_ij
        scalar_emtensor.S = em4phi * np.einsum('xjk,xjk->x', bar_gamma_UU, scalar_emtensor.Sij)        
            
        return scalar_emtensor

    def get_matter_rhs(self, r, bssn_vars, bssn_d1, bssn_d2, bssn_rhs, grid, background):

        assert self.matter_vars_set, 'Matter vars not set'        
        
        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, bssn_vars.h_LL, bssn_d1.h_LL, background)
        
        # \bar \Gamma^i_jk
        bar_chris = get_bar_christoffel(r, Delta_ULL, background) 
        
        em4phi = np.exp(-4.0*bssn_vars.phi)    
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

        dudt =  bssn_vars.lapse * self.v #this is just lapse times first derivative of du/dt (=v)
        dvdt =  (bssn_vars.lapse * bssn_vars.K * self.v 
                 + 2.0 * bssn_vars.lapse * em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, bssn_d1.phi, self.d1_u)
                 +       bssn_vars.lapse * em4phi * np.einsum('xij,xij->x', bar_gamma_UU, self.d2_u)
                 +                         em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, bssn_d1.lapse, self.d1_u)
                 -       bssn_vars.lapse * em4phi * np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, self.d1_u))

        # Add mass term
        dvdt += - bssn_vars.lapse * self.dVdu(self.u)

        ########################################################################################################
        ########################################################################################################
        # MODIFICATION FOR SCALAR GAUS BONNET PART (extra term to dvdt)

        # Calculates the gauss bonnet scalar
        L_GB = compute_L_GB(bssn_vars, bssn_rhs, bssn_d1, bssn_d2, grid, background)[0] #idx=0 corresponds with L_GB

        chi = np.exp(-4.0* bssn_vars.phi) 
        
        # The value of the coupling is in the function 'd1_Lambda_d1_u'
        lambda_GB = 0.05
        chi = np.exp(-4.0* bssn_vars.phi) 
        chi0 = 0.15
        function_of_lambda= (1)/(1+np.exp(-100*(chi-chi0)))
    
        def d1_Lambda_d1_u(lambda_GB):
            return (lambda_GB*function_of_lambda)

        dvdt +=  d1_Lambda_d1_u(lambda_GB) * bssn_vars.lapse * L_GB
        ########################################################################################################
        ########################################################################################################
        
        # Now advection
        dudt   += np.einsum('xj,xj->x', background.inverse_scaling_vector * bssn_vars.shift_U,   self.advec_u)
        dvdt   += np.einsum('xj,xj->x', background.inverse_scaling_vector * bssn_vars.shift_U,   self.advec_v)
        
        return dudt, dvdt
    
    # Set the matter vars and their derivs from the full state vector
    def set_matter_vars(self, state_vector, bssn_vars : BSSNVars, grid : Grid) :
         
        (self.u, self.v) = state_vector[self.idx_u], state_vector[self.idx_v]
        
        # get the derivatives of u needed for the evolution
        # need to get rid of the dependence on the background here somewhow...
        self.d1_u = np.zeros([grid.N, SPACEDIM])
        self.d2_u = np.zeros([grid.N, SPACEDIM, SPACEDIM]) 
        d1_state = grid.get_first_derivative(state_vector, [self.idx_u] )
        self.d1_u[:,i_x1] = d1_state[self.idx_u]
        d2_state = grid.get_second_derivative(state_vector, [self.idx_u])
        self.d2_u[:,i_x1,i_x1] = d2_state[self.idx_u]

        # Derivatives of v
        self.d1_v = np.zeros([grid.N, SPACEDIM])
        d1_state_v = grid.get_first_derivative(state_vector, [self.idx_v] )
        self.d1_v[:,i_x1] = d1_state_v[self.idx_v]
        
        # Advective derivs
        advec_state = grid.get_advection(state_vector, bssn_vars.shift_U[:,i_x1] >= 0, self.indices)
        self.advec_u = np.zeros([grid.N, SPACEDIM])
        self.advec_v = np.zeros([grid.N, SPACEDIM])
        self.advec_u[:,i_x1], self.advec_v[:,i_x1] = advec_state[self.indices]
        
        self.matter_vars_set = True
        