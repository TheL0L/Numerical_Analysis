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


if __name__ == '__main__':
    mat = np.array([
        [1, -1, -2],
        [2, -3, -5],
        [-1, 3, 5],
    ])

    