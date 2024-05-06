import os
import sys
import numpy as np
import importlib

class TinyMPC:
    def __init__(self):
        self.nx = 0 # number of states
        self.nu = 0 # number of control inputs
        self.N = 0 # number of knotpoints in the horizon
        self.A = [] # state transition matrix
        self.B = [] # control matrix
        self.Q = [] # state cost matrix (diagonal)
        self.R = [] # input cost matrix (digaonal)
        self.rho = 0
        self.x_min = [] # lower bounds on state
        self.x_max = [] # upper bounds on state
        self.u_min = [] # lower bounds on input
        self.u_max = [] # upper bounds on input

        # Import tinympc pybind extension
        self.ext = importlib.import_module("tinympc.ext_tinympc")

        self._tinytype = np.float32
        self._infty = 1e17 # TODO: make this max system value
        
        self._solver = None # Solver that stores its own settings, cache, and problem vars/workspace
        self.settings = None # Local settings
    
    
    def update_settings(self, **kwargs):
        assert self.settings is not None
        
        if 'abs_pri_tol' in kwargs:
            self.settings.abs_pri_tol = kwargs.pop('abs_pri_tol')
        if 'abs_dua_tol' in kwargs:
            self.settings.abs_dua_tol = kwargs.pop('abs_dua_tol')
        if 'max_iter' in kwargs:
            self.settings.max_iter = kwargs.pop('max_iter')
        if 'check_termination' in kwargs:
            self.settings.check_termination = kwargs.pop('check_termination')
        if 'en_state_bound' in kwargs:
            self.settings.en_state_bound = 1 if kwargs.pop('en_state_bound') else 0
        if 'en_input_bound' in kwargs:
            self.settings.en_input_bound = 1 if kwargs.pop('en_input_bound') else 0

        if self._solver is not None:
            self._solver.update_settings(self.settings)        
        
        

    # Setup the problem data and solver options
    def setup(self, A, B, Q, R, N, rho=1.0,
        x_min=None, x_max=None, u_min=None, u_max=None, **settings):
        """Instantiate necessary algorithm variables and parameters
        
        :param A (np.ndarray): State transition matrix of the linear system, size nx x nx
        :param B (np.ndarray): Input matrix of the linear system, size nx x nu
        :param Q (np.ndarray): Stage cost for state, must be diagonal and positive semi-definite, size nx x nx
        :param R (np.ndarray): Stage cost for input, must be diagonal and positive definite, size nu x nu
        :param rho (int, optional): Penalty term used in ADMM, default 1
        :param x_min (list[float] or None, optional): Lower bound state constraints of the same length as nx, default None
        :param x_max (list[float] or None, optional): Upper bound state constraints of the same length as nx, default None
        :param u_min (list[float] or None, optional): Lower bound input constraints of the same length as nu, default None
        :param u_max (list[float] or None, optional): Upper bound input constraints of the same length as nu, default None
        :params settings: Dictionary of optional settings
            :param abs_pri_tol (float): Solution tolerance for primal variables
            :param abs_dua_tol (float): Solution tolerance for dual variables
            :param max_iter (int): Maximum number of iterations before returning
            :param check_termination (int): Number of iterations to skip before checking termination
            :param en_state_bound (bool): Enable or disable bound constraints on state
            :param en_input_bound (bool): Enable or disable bound constraints on input
        """
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]
        assert Q.shape[0] == Q.shape[1]
        assert A.shape[0] == Q.shape[0]
        assert R.shape[0] == R.shape[1]
        assert B.shape[1] == R.shape[0]
        self.A = np.array(A, order="F") # order=F for compatibility with eigen's column-major storage when using pybind
        self.B = np.array(B, order="F")
        self.Q = np.array(Q, order="F")
        self.R = np.array(R, order="F")
        self.rho = rho
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.N = N
        self.x_min = x_min if x_min is not None else np.array([-self._infty]*self.nx)
        self.x_max = x_max if x_max is not None else np.array([self._infty]*self.nx)
        self.u_min = u_min if u_min is not None else np.array([-self._infty]*self.nu)
        self.u_max = u_max if u_max is not None else np.array([self._infty]*self.nu)
        assert len(self.x_min.shape) == 1
        assert len(self.x_max.shape) == 1
        assert len(self.u_min.shape) == 1
        assert len(self.u_max.shape) == 1
        assert len(self.x_min) == self.nx
        assert len(self.x_max) == self.nx
        assert len(self.u_min) == self.nu
        assert len(self.u_max) == self.nu

        self.settings = self.ext.TinySettings() # instantiate local settings (settings known only to the python interface)
        self.ext.tiny_set_default_settings(self.settings) # set local settings to default defined by C++ implementation
        self.update_settings(**settings) # change local settings based on arguments available to the interface

        self._solver = self.ext.TinySolver(self.A, self.B, self.Q, self.R, self.rho,
                                           self.nx, self.nu, self.N,
                                           self.x_min, self.x_max, self.u_min, self.u_max,
                                           self.settings
        )

    def solve(self):
        self._solver.solve()