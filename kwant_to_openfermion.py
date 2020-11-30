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

def spin_system_to_FermionicOperator(sys):
    
    n_spin = 2
    
    ham = openfermion.FermionOperator()

    #on site terms
    for lat_ix in sys.id_by_site.values():
        val = sys.hamiltonian(lat_ix, lat_ix)
        for spin_ix1 in range(n_spin):
            for spin_ix2 in range(n_spin):
                ham = ham + openfermion.FermionOperator(f'{n_spin*lat_ix+spin_ix1}^ {n_spin*lat_ix+spin_ix2}', val[spin_ix1, spin_ix2])

    #hopping terms
    for edge in range(sys.graph.num_edges):
        lat_ix1 = sys.graph.head(edge)
        lat_ix2 = sys.graph.tail(edge)
        val = sys.hamiltonian(lat_ix1, lat_ix2)
        for spin_ix1 in range(n_spin):
            for spin_ix2 in range(n_spin):
                ham = ham + openfermion.FermionOperator(f'{n_spin*lat_ix1+spin_ix1}^ {n_spin*lat_ix2+spin_ix2}', val[spin_ix1, spin_ix2])

        
    return(ham)