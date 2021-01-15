import openfermion
import kwant
from itertools import combinations

class Indexer:
    '''
    An object that matches lattice and spin indices to
    integer indices to be used by openfermion operators.
    '''

    def __init__(self):
        self.elements = []
        self._index_by_element = {}
        self._current_index = 0

    def index(self, el):
        '''
        Return an integer index for el.
        '''

        #If the element was not indexed already, create a new index
        if not el in self.elements:
            self.elements.append(el)
            self._index_by_element[el] = self._current_index
            self._current_index += 1

        return self._index_by_element[el]

    def element(self, ix):
        '''
        Return the element indexed by ix.
        '''
        return self.elements[ix]


def _single_term_to_FermionOperator(val, lat_ix1, lat_ix2, ind):
    '''
    Export single term of the hamiltonian to openfermion.

    Parameters
    ----------
    val: number or 2D array

    Returns
    ----------
    op: openfermion.FermionOperator
    '''
    try:
        n_spin1 = val.shape[0]
        n_spin2 = val.shape[1]

        op = openfermion.FermionOperator()
        for spin_ix1 in range(n_spin1):
            for spin_ix2 in range(n_spin2):
                ix1 = ind.index((lat_ix1, spin_ix1, n_spin1))
                ix2 = ind.index((lat_ix2, spin_ix2, n_spin2))
                op += openfermion.FermionOperator(f'{ix1}^ {ix2}', val[spin_ix1, spin_ix2])

        return op

    except:
        try:
            ix1 = ind.index((lat_ix1, 0, 1))
            ix2 = ind.index((lat_ix2, 0, 1))
            return openfermion.FermionOperator(f'{ix1}^ {ix2}', val)
        except:
            raise ValueError(f'''
            Cannot construct fermionic operator with
            indices {lat_ix1}, {lat_ix2}, value {val}''')

def system_to_FermionOperator(sys, return_indexer = False):
    '''
    Export the hamiltonian of a kwant system to openfermion.

    Parameters
    ----------
    sys: kwant.system.FiniteSystem or kwant.system.InfiniteSystem
    return_indexer: bool

    Returns
    ----------
    ham: openfermion.FermionOperator
        The hamiltonian of sys as an openfermion object.
    indexer: kwant_to_openfermion.Indexer
        An object that matched the interger indices used by the
        openfermion operator to fermionic modes.
        Fermionic modes are described as tuples containing
        (site index, spin state index, 1/total_spin).
    '''
    #TO DO? Site indices are stored as integers.
    #Perhaps tuples/lattice objects would be better?

    if not isinstance(sys, kwant.system.System):
        raise TypeError(f'Expecting an instance of System, got {type(sys)}')

    ham = openfermion.FermionOperator()
    ind = Indexer()

    #on site terms
    for lat_ix in sys.id_by_site.values():
        val = sys.hamiltonian(lat_ix, lat_ix)
        ham += _single_term_to_FermionOperator(val, lat_ix, lat_ix, ind)

    #hopping terms
    for edge in range(sys.graph.num_edges):
        lat_ix1 = sys.graph.head(edge)
        lat_ix2 = sys.graph.tail(edge)
        val = sys.hamiltonian(lat_ix1, lat_ix2)
        ham += _single_term_to_FermionOperator(val, lat_ix1, lat_ix2, ind)

    if return_indexer:
        return ham, ind
    return ham

def hubbard_interaction(U, ind):
    '''
    Return onsite interaction of the form U*n_{i, sigma}n_{i, sigma'}
    for the sytem as an openfermion object.
    Parameters
    ----------
    U: numeric
        Interaction strength.
    ind: kwant_to_openfermion.Indexer
        Matching fermionic modes to integer indices to be used.
    
    Returns
    ----------
    int_ham: openfermion.FermionOperator
        The hamiltonian describing the interaction.
    '''
    
    int_ham = openfermion.FermionOperator()
    
    #List of site indices and number of fermionic modes on each site
    sites = set([(i[0], i[2]) for i in ind.elements])
    for lat_ix, n_spin in sites:
        for spin_ix1, spin_ix2 in combinations(range(n_spin), 2):
            ix1 = ind.index((lat_ix, spin_ix1, n_spin))
            ix2 = ind.index((lat_ix, spin_ix2, n_spin))
            int_ham += openfermion.FermionOperator(f'{ix1}^ {ix1} {ix2}^ {ix2}', U)
    
    return int_ham