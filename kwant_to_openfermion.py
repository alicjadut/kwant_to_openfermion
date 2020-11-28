import openfermion
import kwant


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