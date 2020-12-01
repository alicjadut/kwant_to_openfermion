import openfermion
import kwant
import numpy
import tinyarray

def _index(lattice_index, spin_index, n_spin):
    '''
    Create an integer index based on lattice and spin indices, following the openfermion convension.
    '''
    return lattice_index*n_spin + spin_index

def _check_dimension(value, n):
    '''
    Check if value is a numeric array of size n x n (or a scalar in case of n=1)
    '''
    #if value is an array
    if isinstance(value, (numpy.ndarray, tinyarray.ndarray_complex, tinyarray.ndarray_float, tinyarray.ndarray_int)):
        if value.shape == (n, n):
            return
        raise ValueError(f'Expected an array of shape ({n},{n}), got {value.shape}')
    #numeric value accepted only for n=1
    if isinstance(value, (int, float, complex)):
        if n == 1:
            return
        raise ValueError(f'Got a scalar, please convert to  an ({n},{n}) array')
    #Else: wrong type
    raise TypeError(f'Expected a number or a numeric array, got {type(value)}')


def _single_term_to_FermionOperator(val, lat_ix1, lat_ix2, n_spin):
    '''
    Export single term of the hamiltonian to openfermion.

    Parameters
    ----------
    val: number or array

    Returns
    ----------
    op: openfermion.FermionOperator
    '''
    _check_dimension(val, n_spin)
    if isinstance(val, (int, float, complex)):
        op = openfermion.FermionOperator(f'{lat_ix1}^ {lat_ix2}', val)
    else:
        op = openfermion.FermionOperator()
        for spin_ix1 in range(n_spin):
            for spin_ix2 in range(n_spin):
                ix1 = _index(lat_ix1, spin_ix1, n_spin)
                ix2 = _index(lat_ix2, spin_ix2, n_spin)
                op += openfermion.FermionOperator(f'{ix1}^ {ix2}', val[spin_ix1, spin_ix2])
    return op

def system_to_FermionOperator(sys):
    '''
    Export the hamiltonian of a kwant system to openfermion.

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

    #Get the number of spin states out of the first on-site value
    sample_val = sys.onsites[0][0]
    if isinstance(sample_val, (int, float, complex)):
        n_spin = 1
    elif isinstance(sample_val, (numpy.ndarray, tinyarray.ndarray_complex, tinyarray.ndarray_float, tinyarray.ndarray_int)):
        n_spin = sample_val.shape[0]
    else:
        raise TypeError(f'Expected a number or a numeric array, got {type(sample_val)}')

    ham = openfermion.FermionOperator()

    #on site terms
    for lat_ix in sys.id_by_site.values():
        val = sys.hamiltonian(lat_ix, lat_ix)
        ham += _single_term_to_FermionOperator(val, lat_ix, lat_ix, n_spin)

    #hopping terms
    for edge in range(sys.graph.num_edges):
        lat_ix1 = sys.graph.head(edge)
        lat_ix2 = sys.graph.tail(edge)
        val = sys.hamiltonian(lat_ix1, lat_ix2)
        ham += _single_term_to_FermionOperator(val, lat_ix1, lat_ix2, n_spin)

    return ham

