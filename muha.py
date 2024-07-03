import numpy as np

def print_matrix(matrix):
    """
    Prints a matrix with elements aligned in columns.

    Parameters:
    matrix (np.ndarray): The matrix to print.
    """
    # Find the maximum width of the elements
    max_width = max(len(str(element)) for row in matrix for element in row)

    # Print each row with elements aligned to the maximum width
    for row in matrix:
        print("  ".join(f"{str(element):>{max_width}}" for element in row))

    print("\n")


def random_matrix(N, M, low=0, high=10):
    """
    Generates a random matrix of size N x M with integer values between low and high (inclusive).
    
    Parameters:
    N (int): Number of rows.
    M (int): Number of columns.
    low (int): Minimum value (inclusive) for random integers. Default is 0.
    high (int): Maximum value (inclusive) for random integers. Default is 10.
    
    Returns:
    np.array: N x M matrix with random integers.
    """
    return np.random.randint(low, high + 1, size=(N, M))


def determinant(matrix):
    """
    Computes the determinant of a square matrix.

    Parameters:
    matrix (np.ndarray): The square matrix for which the determinant is to be computed.

    Returns:
    float: The determinant of the matrix.

    Raises:
    ValueError: If the input is not a square matrix.
    """
    # Check if the input is a square matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Matrix must be square (n x n)')

    # Base case for 1x1 matrix
    if matrix.shape == (1, 1):
        return matrix[0, 0]

    # Base case for 2x2 matrix
    if matrix.shape == (2, 2):
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    # Recursive case for larger matrices
    det = 0
    for col in range(matrix.shape[1]):
        sub_matrix = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)
        cofactor = ((-1) ** col) * matrix[0, col] * determinant(sub_matrix)
        det += cofactor

    return det


def get_matrix_norm(matrix: np.ndarray) -> int:
    """
    Computes the maximum row sum (infinity norm) of a matrix.

    Parameters:
    matrix (np.ndarray): The matrix for which the norm is to be computed.

    Returns:
    int: The maximum row sum of the matrix.
    """
    size = len(matrix)
    max_row = 0
    for row in range(size):
        sum_row = 0
        for col in range(size):
            sum_row += abs(matrix[row][col])
        if sum_row > max_row:
            max_row = sum_row
    return max_row


def get_matrix_cond(matrix: np.ndarray, inverse_matrix: np.ndarray) -> int:
    """
    Computes the condition number of a matrix.

    Parameters:
    matrix (np.ndarray): The original matrix.
    inverse_matrix (np.ndarray): The inverse of the original matrix.

    Returns:
    int: The condition number of the matrix.
    """
    return get_matrix_norm(matrix) * get_matrix_norm(inverse_matrix)


def is_square(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is square.

    Parameters:
    matrix (np.ndarray): The matrix to check.

    Returns:
    bool: True if the matrix is square, False otherwise.
    """
    return matrix.shape[0] == matrix.shape[1]


def get_dim_of_square_matrix(matrix: np.ndarray) -> int:
    """
    Get the dimension of a square matrix.

    Parameters:
    matrix (np.ndarray): The matrix whose dimension is to be retrieved.

    Returns:
    int: The dimension of the square matrix.
    """
    return matrix.shape[0]


def row_addition_elementary_matrix(dim: int, target_row: int, addend_row: int, scalar: float = 1.0) -> np.ndarray:
    """
    Create an elementary matrix that adds a multiple of one row to another.

    Parameters:
    dim (int): Dimension of the square matrix.
    target_row (int): Index of the row to be modified.
    addend_row (int): Index of the row to be added.
    scalar (float, optional): Scalar multiple of the addend row. Defaults to 1.0.

    Returns:
    np.ndarray: The elementary matrix for the row addition operation.

    Raises:
    ValueError: If row indices are invalid or if target_row and addend_row are the same.
    """
    if target_row < 0 or addend_row < 0 or target_row >= dim or addend_row >= dim:
        raise ValueError('Invalid row indices.')

    if target_row == addend_row:
        raise ValueError('Source and target rows cannot be the same.')

    elementary_matrix = np.identity(dim)
    elementary_matrix[target_row, addend_row] = scalar

    return np.array(elementary_matrix)


def scalar_multiplication_elementary_matrix(dim: int, row: int, scalar: float) -> np.ndarray:
    """
    Create an elementary matrix that multiplies a row by a scalar.

    Parameters:
    dim (int): Dimension of the square matrix.
    row (int): Index of the row to be scaled.
    scalar (float): Scalar to multiply the row by.

    Returns:
    np.ndarray: The elementary matrix for the scalar multiplication operation.

    Raises:
    ValueError: If row index is invalid or if scalar is zero.
    """
    if row < 0 or row >= dim:
        raise ValueError('Invalid row index.')

    if scalar == 0:
        raise ValueError('Scalar cannot be zero for row multiplication.')

    elementary_matrix = np.identity(dim)
    elementary_matrix[row, row] = scalar

    return np.array(elementary_matrix)


def get_inverse_elemental_matrices(matrix: np.ndarray) -> list[np.ndarray]:
    """
    Compute the sequence of elementary matrices used to invert a given matrix.

    Parameters:
    matrix (np.ndarray): The square matrix to be inverted.

    Returns:
    list[np.ndarray]: A list of elementary matrices used to invert the input matrix.

    Raises:
    ValueError: If the matrix is singular or not square.
    """
    dim = get_dim_of_square_matrix(matrix)
    elementary_matrices = []

    # iterate over each row
    for i in range(dim):
        if matrix[i, i] == 0:
            raise ValueError('Matrix is singular, cannot find its inverse.')

        # Scale the current row to make the diagonal element 1
        if matrix[i, i] != 1:
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(dim, i, scalar)
            matrix = np.dot(elementary_matrix, matrix)
            elementary_matrices.append(elementary_matrix)

        # Zero out the elements above and below the diagonal
        rows_above_below = list(range(dim))
        rows_above_below.remove(i)
        for j in rows_above_below:
            scalar = -matrix[j, i]
            elementary_matrix = row_addition_elementary_matrix(dim, j, i, scalar)
            matrix = np.dot(elementary_matrix, matrix)
            elementary_matrices.append(elementary_matrix)

    return elementary_matrices


def inverse(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a square matrix using elementary row operations.

    Parameters:
    matrix (np.ndarray): The square matrix to be inverted.

    Returns:
    np.ndarray: The inverse of the input matrix.

    Raises:
    ValueError: If the matrix is not square.
    """
    if not is_square(matrix):
        raise ValueError('Input matrix must be square.')

    inv = np.identity(get_dim_of_square_matrix(matrix))

    # use pivoting?
    elementary_matrices = get_inverse_elemental_matrices(matrix)
    for elementary in elementary_matrices:
        inv = np.dot(elementary, inv)

    return inv


if __name__ == '__main__':
    mat = np.array([
        [1, -1, -2],
        [2, -3, -5],
        [-1, 3, 5],
    ])

    # Step 1: Calculate the max norm (infinity norm) of A
    norm_mat = get_matrix_norm(mat)

    # Step 2: Calculate the inverse of A
    mat_inv = inverse(mat)

    # Step 3: Calculate the max norm of the inverse of A
    norm_inv = get_matrix_norm(mat_inv)

    # Step 4: Compute the condition number
    cond = get_matrix_cond(mat, mat_inv)

    print('Original Matrix:')
    print_matrix(mat)

    print('Inverse Matrix:')
    print_matrix(mat_inv)

    print(f'Max norm of Matrix is {norm_mat}\n')

    print(f'Max norm of Inverse Matrix is {norm_inv}\n')
    
    print(f'Condition Number is {cond}\n')
