import openfermion
import kwant
import numpy
import tinyarray

class Indexer:
    
    def __init__(self):
        self._indexed_elements = []
        self._index_by_element = {}
        self._current_index = 0
        
    def index(self, el):
        '''
        Return an integer index for el.
        '''
        
        #If the element was not indexed already, create a new index
        if not el in self._indexed_elements:
            self._indexed_elements.append(el)
            self._index_by_element[el] = self._current_index
            self._current_index += 1
        
        return self._index_by_element[el]
        
    def element(self, ix):
        '''
        Return the element indexed by ix.
        '''
        return self._indexed_elements[ix]
        

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
            raise ValueError(f'Cannot construct fermionic operator with indices {lat_ix1}, {lat_ix2}, value {val}')

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
        openfermion operator to site and spin indices.
    '''

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

    if(return_indexer):
        return ham, indexer
    return ham
