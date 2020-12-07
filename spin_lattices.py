import numpy as np
import openfermion
import kwant


# Define Pauli matrices
sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
pauli_matrices = [sigma_0, sigma_x, sigma_y, sigma_z]
pauli_names = ['1', 'X', 'Y', 'Z']

#Define tensor products of pauli matrices
pauli_matrices4 = [np.kron(matrix1, matrix2) for matrix1 in pauli_matrices for matrix2 in pauli_matrices]
pauli_names4 = [(name1, name2) for name1 in pauli_names for name2 in pauli_names]


# Transformation from the computational basis to the Pauli basis
## 2x2 matrices
transformation_matrix2 = np.transpose(np.linalg.inv([m.reshape(4) for m in pauli_matrices]))
def _to_pauli_basis_2(m):
    vec = m.reshape(4)
    return np.dot(transformation_matrix2, vec)
    
## 4x4 matrices
transformation_matrix4 = np.transpose(np.linalg.inv([m.reshape(4*4) for m in pauli_matrices4]))
def _to_pauli_basis_4(m):
    vec = m.reshape(4*4)
    return np.dot(transformation_matrix4, vec)

def to_pauli_basis(m):
    try:
        return _to_pauli_basis_2(m)
    except:
        try:
            return _to_pauli_basis_4(m)
        except:
            raise ValueError('Cannot get Pauli coefficients.')


            


def _single_term_to_QubitOperator(val, ix1, ix2):
    
    try:
        dims = val.shape
    except:
        raise ValueError(f'Expected a matrix, got {type(val)}.')
        
    pauli_coefs = to_pauli_basis(val)
    
    op = openfermion.QubitOperator()
    
    #On site terms
    if ix1 == ix2:
        assert dims == (2,2), f'Onsite terms must be 2x2 matrices, got {dims}'
        for name, coef in zip(pauli_names, pauli_coefs):
            if name == '1':
                op += openfermion.QubitOperator('', coef)
            else:
                op += openfermion.QubitOperator(name+str(ix1), coef)
        return op
    
    #Interaction terms
    assert dims == (4,4), f'Onsite terms must be 4x4 matrices, got {dims}'
    for name, coef in zip(pauli_names4, pauli_coefs):
        if name == ('1', '1'):
            op += openfermion.QubitOperator('', coef)
        elif name[0] == '1':
            op += openfermion.QubitOperator(name[1]+str(ix2), coef)
        elif name[1] == '1':
            op += openfermion.QubitOperator(name[0]+str(ix1), coef)
        else:
            op += openfermion.QubitOperator(name[0]+str(ix1)+' '+name[1]+str(ix2), coef)
    return op



def system_to_QubitOperator(sys):
    
    if not isinstance(sys, kwant.system.System):
        raise TypeError(f'Expecting an instance of System, got {type(sys)}')

    ham = openfermion.QubitOperator()

    #on site terms
    for lat_ix in sys.id_by_site.values():
        val = sys.hamiltonian(lat_ix, lat_ix)
        ham += _single_term_to_QubitOperator(val, lat_ix, lat_ix)

    #hopping terms
    for edge in range(sys.graph.num_edges):
        lat_ix1 = sys.graph.head(edge)
        lat_ix2 = sys.graph.tail(edge)
        val = sys.hamiltonian(lat_ix1, lat_ix2)/2
        ham += _single_term_to_QubitOperator(val, lat_ix1, lat_ix2)

    return ham