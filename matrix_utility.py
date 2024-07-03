import numpy as np


def print_matrix(matrix: np.ndarray) -> None:
    """
    Prints the given matrix with elements aligned for better readability.
    
    Parameters:
    matrix (np.ndarray): The matrix to be printed.
    
    Returns:
    None
    """
    max_width = max(len(str(element)) for row in matrix for element in row)
    for row in matrix:
        print("  ".join(f"{str(element):>{max_width}}" for element in row))


def random_matrix(N: int, M: int, low: int = 0, high: int = 10) -> np.ndarray:
    """
    Generates a random matrix of size N x M with integer values between low and high (inclusive).
    
    Parameters:
    N (int): Number of rows.
    M (int): Number of columns.
    low (int): Minimum value (inclusive) for random integers. Default is 0.
    high (int): Maximum value (inclusive) for random integers. Default is 10.
    
    Returns:
    np.ndarray: N x M matrix with random integers.
    """
    return np.random.randint(low, high + 1, size=(N, M))


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


def determinant(matrix: np.ndarray) -> float:
    """
    Calculates the determinant of a given square matrix.
    
    Parameters:
    matrix (np.ndarray): The input square matrix.
    
    Returns:
    float: The determinant of the matrix.
    
    Raises:
    ValueError: If the matrix is not square.
    """
    if not is_square(matrix):
        raise ValueError("Input matrix must be square.")

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


def elementary_matrix_for_row_addition(dim: int, target_row: int, addend_row: int, scalar: float = 1.0) -> np.ndarray:
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
        raise ValueError("Invalid row indices.")

    if target_row == addend_row:
        raise ValueError("Source and target rows cannot be the same.")

    elementary_matrix = np.identity(dim)
    elementary_matrix[target_row, addend_row] = scalar

    return np.array(elementary_matrix)


def elementary_matrix_for_row_swap(dim: int, row_a: int, row_b: int) -> np.ndarray:
    """
    Create an elementary matrix that swaps two rows.

    Parameters:
    dim (int): Dimension of the square matrix.
    row_a (int): Index of the row to be swapped.
    row_b (int): Index of the row to be swapped.

    Returns:
    np.ndarray: The elementary matrix for the row swapping operation.

    Raises:
    ValueError: If row indices are invalid or if row_a and row_b are the same.
    """
    if row_a < 0 or row_b < 0 or row_a >= dim or row_b >= dim:
        raise ValueError("Invalid row indices.")

    if row_a == row_b:
        raise ValueError("Swapped rows cannot be the same.")

    elementary_matrix = np.identity(dim)
    elementary_matrix[[row_a, row_b]] = elementary_matrix[[row_b, row_a]]

    return np.array(elementary_matrix)


def elementary_matrix_for_scalar_multiplication(dim: int, row: int, scalar: float) -> np.ndarray:
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
        raise ValueError("Invalid row index.")

    if scalar == 0:
        raise ValueError("Scalar cannot be zero for row multiplication.")

    elementary_matrix = np.identity(dim)
    elementary_matrix[row, row] = scalar

    return np.array(elementary_matrix)


def get_inverse_elementary_matrices(matrix: np.ndarray) -> list[np.ndarray]:
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

    # Iterate over each row
    for i in range(dim):
        if matrix[i, i] == 0:
            raise ValueError("Matrix is singular, cannot find its inverse.")

        # Scale the current row to normalize the diagonal
        if matrix[i, i] != 1:
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = elementary_matrix_for_scalar_multiplication(dim, i, scalar)
            matrix = np.dot(elementary_matrix, matrix)
            elementary_matrices.append(elementary_matrix)

        # Zero out the elements above and below the diagonal
        rows_above_below = list(range(dim))
        rows_above_below.remove(i)
        for j in rows_above_below:
            scalar = -matrix[j, i]
            elementary_matrix = elementary_matrix_for_row_addition(dim, j, i, scalar)
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
        raise ValueError("Input matrix must be square.")

    inv = np.identity(get_dim_of_square_matrix(matrix))

    # TODO: use pivoting?
    elementary_matrices = get_inverse_elementary_matrices(matrix)
    for elementary in elementary_matrices:
        inv = np.dot(elementary, inv)

    return inv


def max_norm(matrix: np.ndarray) -> float:
    """
    Calculates the max norm (infinity norm) of a given matrix.
    
    Parameters:
    matrix (np.ndarray): The input matrix.
    
    Returns:
    float: The max norm of the matrix.
    """
    # Calculate the sum of absolute values of each row
    row_sums = np.sum(np.abs(matrix), axis=1)
    
    # Find and return the maximum row sum
    return np.max(row_sums)


def condition(matrix: np.ndarray) -> float:
    """
    Calculates the condition value of a given matrix using the max norm (infinity norm).
    
    Parameters:
    matrix (np.ndarray): The input matrix.
    
    Returns:
    float: The condition value of the matrix.
    """
    return max_norm(matrix) * max_norm(inverse(matrix))


def is_diagonally_dominant(matrix: np.ndarray) -> bool:
    """
    Checks if the given matrix is diagonally dominant.
    
    Parameters:
    matrix (np.ndarray): The input matrix.
    
    Returns:
    bool: True if the matrix is diagonally dominant, False otherwise.
    """
    if matrix is None:
        return False

    # Calculate the absolute values of the diagonal elements
    diagonal_elements = np.diag(np.abs(matrix))
    
    # Calculate the row sums without the diagonal elements
    row_sums = np.sum(np.abs(matrix), axis=1) - diagonal_elements
    
    # Check if each diagonal element is greater than the corresponding row sum
    return np.all(diagonal_elements > row_sums)


if __name__ == '__main__':
    mat = np.array([
        [1, -1, -2],
        [2, -3, -5],
        [-1, 3, 5],
    ])

    inv = inverse(mat)

    print('Original matrix:')
    print_matrix(mat)

    print(f'\nDeterminant: {determinant(mat)}')

    print('\nInverse:')
    print_matrix(inv)

    print(f'\nMax-Norm of original matrix: {max_norm(mat)}')
    print(f'Max-Norm of inverse matrix:  {max_norm(inv)}')
    print(f'Condition value:             {condition(mat)}')
    
