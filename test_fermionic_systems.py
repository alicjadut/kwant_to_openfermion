import numpy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)#Ignore the "MUMPS unavailable" warning
import kwant
import openfermion
import fermionic_systems as fs


sigma_0 = numpy.array([[1., 0], [0, 1]])
    
def test_single_term_spinless():
    op1 = fs._single_term_to_FermionOperator(1, 0, 0, fs.Indexer())
    op2 = openfermion.FermionOperator('0^ 0')
    assert op1 == op2

def test_single_term_spin():
    op1 = fs._single_term_to_FermionOperator(sigma_0, 0, 0, fs.Indexer())
    op2 = openfermion.FermionOperator('0^ 0')+openfermion.FermionOperator('1^ 1')
    assert op1 == op2
    
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
        ham += openfermion.FermionOperator(f'{i}^ {i}', t)
    ##Hopping terms
    for i in range(L-1):
        term = openfermion.FermionOperator(f'{i}^ {i+1}', -t)
        ham += term
        ham += openfermion.hermitian_conjugated(term)
   
    #Test if equal
    assert fs.system_to_FermionOperator(syst) == ham

def test_spin_hubbard():
    '''
    This test relies on openfermion.hamiltonians.fermi_hubard.
    '''

    a = 1
    L = 2
    t = 1

    #Build an LxL square lattice in kwant
    lat = kwant.lattice.square(a)
    syst = kwant.Builder()   
    syst[(lat(x, y) for x in range(L) for y in range(L))] = 0 * sigma_0
    ##Hopping terms between neighbours with the same spin
    syst[lat.neighbors()] = -t * sigma_0
    syst = syst.finalized()

    #Build a hamiltonian in openfermion
    ham = openfermion.hamiltonians.fermi_hubbard(L, L, t, 0, spinless=False)
  
    #Test if equal
    assert fs.system_to_FermionOperator(syst) == ham
    
def test_different_spins():
    '''
    Non-interacting fermions with different spins.
    '''
    
    n_sites = 3
    
    lat = kwant.lattice.chain()
    syst = kwant.Builder()
    for i in range(n_sites):
        syst[lat(i)] = numpy.diag(numpy.ones(i+1))
    syst = syst.finalized()
    
    ham = openfermion.FermionOperator()
    for i in range((n_sites*(n_sites+1))//2):
        ham += openfermion.FermionOperator(f'{i}^ {i}')
        
    assert fs.system_to_FermionOperator(syst) == ham
    
def test_different_spins_hopping():
    '''
    1 fermionic mode at site 0, 2 fermionic modes at site 1
    + hopping between sites
    '''
    lat = kwant.lattice.chain()
    syst = kwant.Builder()
    syst[lat(0)] = 2.
    syst[lat(1)] = 3. * sigma_0
    syst[lat(0), lat(1)] = numpy.array([[1., -1.]])
    syst = syst.finalized()
    
    ham = \
    openfermion.FermionOperator('0^ 0', 2.) + \
    openfermion.FermionOperator('1^ 1', 3.) + \
    openfermion.FermionOperator('2^ 2', 3.) + \
    openfermion.FermionOperator('0^ 1', 1.) + \
    openfermion.FermionOperator('1^ 0', 1.) + \
    openfermion.FermionOperator('2^ 0', -1.) + \
    openfermion.FermionOperator('0^ 2', -1.)
    
    assert fs.system_to_FermionOperator(syst) == ham
    

