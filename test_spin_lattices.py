import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)#Ignore the "MUMPS unavailable" warning
import kwant
import openfermion
import spin_lattices as sl

def test_pauli_basis():
    assert np.array_equal(sl.to_pauli_basis(sl.sigma_x),[0,1,0,0])
    assert np.array_equal(sl.to_pauli_basis(sl.sigma_y + 2*sl.sigma_z),[0,0,1,2])
    
def test_single_onsite_term():
    assert sl._single_term_to_QubitOperator(sl.sigma_x, 0, 0) == openfermion.QubitOperator('X0')
    
def test_single_interaction_term():
    assert sl._single_term_to_QubitOperator(5*np.kron(sl.sigma_x, sl.sigma_z), 0, 1) == openfermion.QubitOperator('X0 Z1', 5)