import tinyarray
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)#Ignore the "MUMPS unavailable" warning
from kwant_to_openfermion import *
    
def test_chain():
    
    a = 1
    L = 3
    t = 1

    #Build a chain of length L in kwant
    lat = kwant.lattice.chain(a)
    syst = kwant.Builder()   
    ##On-site terms
    syst[(lat(x) for x in range(L))] = t
    ##Hopping terms
    syst[lat.neighbors()] = -t
    syst = syst.finalized()

    #Build a hamiltonian in openfermion
    ham = openfermion.FermionOperator()
    ##On-site terms
    for i in range(L):
        ham+=openfermion.FermionOperator(f'{i}^ {i}', t)
    ##Hopping terms
    for i in range(L-1):
        term = openfermion.FermionOperator(f'{i}^ {i+1}', -t)
        ham+= term
        ham+= openfermion.hermitian_conjugated(term)
        
    #Test if equal
    assert system_to_FermionOperator(syst) == ham
    
def test_spin_hubbard():
    
    a = 1
    L = 2
    t = 1
    
    sigma_0 = tinyarray.array([[1, 0], [0, 1]])

    #Build an LxL square lattice in kwant
    lat = kwant.lattice.square(a)
    syst = kwant.Builder()   
    syst[(lat(x, y) for x in range(L) for y in range(L))] = 0 * sigma_0
    ##Hopping terms between neighbours with the same spin
    syst[lat.neighbors()] = -t * sigma_0
    syst = syst.finalized()

    #Build a hamiltonian in openfermion
    ham = openfermion.hamiltonians.fermi_hubbard(L, L, t, 0, spinless = False)
        
    #Test if equal
    assert spin_system_to_FermionOperator(syst) == ham
    
    
    