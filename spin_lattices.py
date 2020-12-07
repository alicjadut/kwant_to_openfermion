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
