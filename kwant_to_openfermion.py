import openfermion
import kwant
import numpy
import tinyarray


def system_to_FermionOperator(sys):
    '''
    Export the hamiltonian of a kwant system to openfermion.
    Currently doesn't support spin systems.
    
    Parameters
    ----------
    sys: kwant.system.FiniteSystem or kwant.system.InfiniteSystem
    
    Returns
    ----------
    ham: openfermion.FermionOperator
        The hamiltonian of sys as an openfermion object.
    '''
    
    if not isinstance(sys, kwant.system.System):
        raise TypeError(f'Expecting an instance of System, got {type(sys)}')
    
    ham = openfermion.FermionOperator()

    #on site terms
    for ix in sys.id_by_site.values():
        val = sys.hamiltonian(ix, ix)
        if not isinstance(val, (int, float, complex)):
            raise ValueError('Currently only supporting numeric on-site terms.')
        ham = ham + openfermion.FermionOperator(f'{ix}^ {ix}', val)

    #hopping terms
    for edge in range(sys.graph.num_edges):
        ix1 = sys.graph.head(edge)
        ix2 = sys.graph.tail(edge)
        val = sys.hamiltonian(ix1, ix2)
        if not isinstance(val, (int, float, complex)):
            raise ValueError('Currently only supporting numeric hopping terms.')
        ham = ham + openfermion.FermionOperator(f'{ix1}^ {ix2}', val)
        
    return(ham)

def _index(lattice_index, spin_index, n_spin):
    '''
    Create an integer index based on lattice and spin indices, following the openfermion convension.
    '''
    return(lattice_index*n_spin + spin_index)

def _check_dimension(value, n):
    '''
    Check if value is a numeric array of size n x n (or a scalar in case of n=1)
    '''
    #if value is an array
    if isinstance(value, (numpy.ndarray, tinyarray.ndarray_complex, tinyarray.ndarray_float, tinyarray.ndarray_int)):
        if value.shape == (n, n):
            return
        else:
            raise ValueError(f'Expected an array of shape ({n},{n}), got {value.shape}')
    #numeric value accepted only for n=1
    if isinstance(value, (int, float, complex)):
        if n == 1:
            return
        else:
            raise ValueError(f'Got a scalar, please convert to  an ({n},{n}) array')
    #Else: wrong type
    raise TypeError(f'Expected a number or a numeric array, got {type(value)}')


def spin_system_to_FermionOperator(sys):
    
    n_spin = 2
    
    ham = openfermion.FermionOperator()

    #on site terms
    for lat_ix in sys.id_by_site.values():
        val = sys.hamiltonian(lat_ix, lat_ix)
        _check_dimension(val, n_spin)
        for spin_ix1 in range(n_spin):
            for spin_ix2 in range(n_spin):
                ix1 = _index(lat_ix, spin_ix1, n_spin)
                ix2 = _index(lat_ix, spin_ix2, n_spin)
                ham = ham + openfermion.FermionOperator(f'{ix1}^ {ix2}', val[spin_ix1, spin_ix2])

    #hopping terms
    for edge in range(sys.graph.num_edges):
        lat_ix1 = sys.graph.head(edge)
        lat_ix2 = sys.graph.tail(edge)
        val = sys.hamiltonian(lat_ix1, lat_ix2)
        _check_dimension(val, n_spin)
        for spin_ix1 in range(n_spin):
            for spin_ix2 in range(n_spin):
                ix1 = _index(lat_ix1, spin_ix1, n_spin)
                ix2 = _index(lat_ix2, spin_ix2, n_spin)
                ham = ham + openfermion.FermionOperator(f'{ix1}^ {ix2}', val[spin_ix1, spin_ix2])
    return(ham)