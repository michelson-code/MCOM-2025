import numpy as np
from scipy.linalg import dft
# importar Warnigs para evitar mensagem de erro ao 
# ignorar operações com matrizes complexas!
import warnings
try:
    from numpy import ComplexWarning
except ImportError:
    from numpy.exceptions import ComplexWarning

# Operacoes com vetores

### Produto escalar-vetor
def scalar_vec_real(a,x,check_input=True):
    '''
    Compute the product of a scalar a and vector x, where
    a is real and x is in R^N.

    The code uses a simple "for" to iterate on the array.

    input
    -----------------
    a: scalar
        Real number

    x: 1D array
       Vector with N elements.

    returns
    ------------------
    y: 1D array
       Vector with N elements equal the product between a and x.

    '''
    if check_input is True:
        assert isinstance(a, (float, int)), 'a must be a scalar'
        assert isinstance(x, (np.ndarray)), 'x must be a numpy array'
        
    x = np.asarray(x, dtype=float)
    
    if check_input is True:
        assert x.ndim == 1, 'x must have ndim = 1'

    result = np.zeros_like(x)
    for i in range(x.size):
        result[i] = a*x[i]

    return result

def scalar_vec_complex(a, x, check_input=True):
    '''
    Compute the dot product of a is a complex number and x
    is a complex vector.

    Parameters
    ----------
    a : scalar
        Complex number.

    x : array 1D
        Complex vector with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Product of a and x.
    '''
    if check_input is True:
        assert isinstance(a, (complex, float, int)), 'a may be complex or scalar'
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'

    # Code here

    result_real = scalar_vec_real(a.real, x.real, check_input=False)
    result_real -= scalar_vec_real(a.imag, x.imag, check_input=False)
    result_imag = scalar_vec_real(a.real, x.imag, check_input=False)
    result_imag += scalar_vec_real(a.imag, x.real, check_input=False)

    result = result_real + 1j*result_imag

    return result

### Dot product
def dot_real(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of R^N. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    # Check input here
    
    if check_input:
        assert len(x) == len(y), 'Numero de elementos em x é diferente de numero de elementos em y'
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
    else:
        pass

    # Code here    
    N = len(x) # lembrar que o N de x e y deve ser igual, fazer o acert
    result = 0
    
    for i in range(0, N):
        result += x[i]*y[i]


    return result


def dot_complex(x, y, check_input=True):
    '''
    Compute the dot product of x and y, where
    x, y are elements of C^N.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with N elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : scalar
        Dot product of x and y.
    '''
    # Check input here

    if check_input:
        assert len(x) == len(y), 'Numero de elementos em x é diferente de numero de elementos em y'
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
    else:
        pass
    # Complete here

    c_R  = dot_real(x.real, y.real)
    c_R -= dot_real(x.imag, y.imag)
    c_I  = dot_real(x.real, y.imag)
    c_I += dot_real(x.imag, y.real)
    result = c_R + 1j*c_I
    return result

# Outer product
def outer_real_simple(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code uses a simple "for" to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    # Check input here

    if check_input:
        # verifica se são vetores 1D
        assert x.ndim == 1 and y.ndim ==1, "x and y must be 1-dimensional arrays"
        # Verifica se x e y são arrays numpy
        assert isinstance(x, np.ndarray) or  isinstance(y, np.ndarray), TypeError("x and y must be numpy arrays")
        # Verifica se os elementos são reais (ignorando partes imaginárias)
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            x = x.real
            y = y.real
    else:
        x = x.real
        y = y.real
    # Complete here
    N = len(x)
    M = len(y)
    result = np.zeros(shape=(N, M))

    for i in range(0, N):
        for j in range(0, M):
            result[i,j] = x[i]*y[j]

    return result


def outer_real_row(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code use a single for to compute the rows of 
    the resultant matrix as a scalar-vector product.

    This code uses the function 'scalar_vec_real'.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    # Check input here
    if check_input:
        # verifica se são vetores 1D
        assert x.ndim == 1 and y.ndim ==1, "x and y must be 1-dimensional arrays"
        # Verifica se x e y são arrays numpy
        assert isinstance(x, np.ndarray) or  isinstance(y, np.ndarray), TypeError("x and y must be numpy arrays")
        # Verifica se os elementos são reais (ignorando partes imaginárias)
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            x = x.real
            y = y.real
    else:
        x = x.real
        y = y.real
    # Complete here

    N = len(x)
    M = len(y)
    result = np.zeros(shape=(N, M))
    for i in range(0, N):
        result[i,:] = x[i]*y[:]
    return result


def outer_real_column(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where
    x in R^N and y in R^M. The imaginary parts are ignored.

    The code use a single for to compute the columns of 
    the resultant matrix as a scalar-vector product.

    This code uses the function 'scalar_vec_real'.

    Parameters
    ----------
    x, y : arrays 1D
        Vectors with real elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 2d
        Outer product of x and y.
    '''
    # Check input here
    if check_input:
        # verifica se são vetores 1D
        assert x.ndim == 1 and y.ndim ==1, "x and y must be 1-dimensional arrays"
        # Verifica se x e y são arrays numpy
        assert isinstance(x, np.ndarray) or  isinstance(y, np.ndarray), TypeError("x and y must be numpy arrays")
        # Verifica se os elementos são reais (ignorando partes imaginárias)
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            x = x.real
            y = y.real
    else:
        x = x.real
        y = y.real
    # Complete here

    N = len(x)
    M = len(y)
    result = np.zeros(shape=(N, M))
    for j in range(0, M):    
        result[:, j] = x[:]*y[j]


    return result


def outer_complex(x, y, check_input=True, function='simple'):
    '''
    Compute the outer product of x and y, where x and y are complex vectors.

    Parameters
    ----------
    x, y : 1D arrays
        Complex vectors.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Defines the outer_real function to be used. The possible
        values are 'simple', 'row' and 'column'. 

    Returns
    -------
    result : 2D array
        Outer product of x and y.
    '''
    x = np.asarray(x, dtype=complex)
    y = np.asarray(y, dtype=complex)

    if check_input:
        # Verifica se function é string
        if not isinstance(function, str):
            raise TypeError("function parameter must be a string")

        if function  not in ['simple', 'row', 'column']:
            raise ValueError(f"function must be one of simple, row, column! Got {function}.")

    else:
        
        function = 'row'

    outer_real = {
        'simple' : outer_real_simple,
        'row' : outer_real_row,
        'column' : outer_real_column
    }
    A_real = outer_real[function](x.real, y.real)
    A_real -= outer_real[function](x.imag, y.imag)
    A_imag = outer_real[function](x.real, y.imag)
    A_imag += outer_real[function](x.imag, y.real)

    # use the syntax outer_real[function] to specify the
    # the outer_real_* function.
    result = A_real + 1j*A_imag

    return result

# Hadamard product
def hadamard_real(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be real vectors or matrices having the same shape.
    The imaginary parts are ignored.

    The code uses a simple doubly nested loop to iterate on the arrays.

    Parameters
    ----------
    x, y : arrays
        Real vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''
    with warnings.catch_warnings():
            warnings.simplefilter("ignore", ComplexWarning)
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

    if check_input:
        # Check if x and y are numpy arrays
        assert  isinstance(x, np.ndarray) or  isinstance(y, np.ndarray), "Inputs x and y must be numpy arrays."
        # Check if x and y have the same shape
        assert x.shape == y.shape,  ValueError("Inputs x and y must have the same shape.")

    else:
        pass
    # Check if x and y are real
        

    
    if x.ndim == 1:  # Vector
        N = x.shape[0]
        result = np.zeros(N)
        for i in range(0, N):
            result[i] = x[i] * y[i]
    elif x.ndim == 2:  # Matrix
        N = x.shape[0]
        M = x.shape[1]
        result = np.zeros(shape=(N, M))
        for i in range(0, N):
            for j in range(0, M):
                result[i, j] = x[i, j] * y[i, j]
    else:
        raise ValueError("Inputs x and y must be 1D or 2D arrays.")

    return result


def hadamard_complex(x, y, check_input=True):
    '''
    Compute the Hadamard (or entrywise) product of x and y, where
    x and y may be complex vectors or matrices having the same shape.

    Parameters
    ----------
    x, y : arrays
        Complex vectors or matrices having the same shape.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array
        Hadamard product of x and y.
    '''

    if check_input:
        assert np.shape(x) == np.shape(y), "Inputs x and y must have the same shape."

    C_R  = hadamard_real(x.real, y.real)
    C_R -= hadamard_real(x.imag, y.imag)
    C_I  = hadamard_real(x.real, y.imag)
    C_I += hadamard_real(x.imag, y.real)
    result = C_R + 1j*C_I

    return result

## Operations with matrix

# Matrix-vector product

def matvec_real_simple(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code uses a simple doubly nested "for" to iterate on the arrays.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''

    N, M = np.shape(A)
    A = A.real
    x = x.real
    
    if check_input:
        assert A.ndim==2, 'A must be array 2D.'
        assert x.ndim ==1, 'x must be array 1D'
        assert M == x.shape[0], f"Matrix columns ({M}) must match vector length ({x.shape[0]})"
        assert isinstance(A, np.ndarray) and isinstance(x, np.ndarray), TypeError("Both A and x must be numpy arrays")
    else:
        pass
    
    y = np.zeros(N)
    for i in range(0, N):
        for j in range(0, M):
            y[i] += A[i,j]*x[j]
    result = y
    return result


def matvec_real_dot(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code replaces a for by a dot product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    N, M = np.shape(A)
    A = A.real
    x = x.real
    
    if check_input:
        assert A.ndim==2, 'A must be array 2D.'
        assert x.ndim ==1, 'x must be array 1D'
        assert M == x.shape[0], f"Matrix columns ({M}) must match vector length ({x.shape[0]})"
        assert isinstance(A, np.ndarray) and isinstance(x, np.ndarray), TypeError("Both A and x must be numpy arrays")
    else:
        pass
    result = np.zeros(N)
    for i in range(0, N):
        result[i] = dot_real(A[i,:], x[:])
    

    return result


def matvec_real_columns(A, x, check_input=True):
    '''
    Compute the matrix-vector product of A and x, where
    A in R^NxM and x in R^M. The imaginary parts are ignored.

    The code replaces a for by a scalar-vector product.

    Parameters
    ----------
    A : array 2D
        NxM matrix with real elements.

    x : array 1D
        Real vector witn M elements.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : array 1D
        Product of A and x.
    '''
    N, M = np.shape(A)
    A = A.real
    x = x.real
    
    if check_input:
        assert A.ndim==2, 'A must be array 2D.'
        assert x.ndim ==1, 'x must be array 1D'
        assert M == x.shape[0], f"Matrix columns ({M}) must match vector length ({x.shape[0]})"
        assert isinstance(A, np.ndarray) and isinstance(x, np.ndarray), TypeError("Both A and x must be numpy arrays")
    else:
        pass
    result = np.zeros(N)
    for j in range(0, M):
        result[:] += scalar_vec_real(x[j], A[:,j])

    return result


def matvec_complex(A, x, check_input=True, function='dot'):
    '''
    Compute the matrix-vector product of an NxM matrix A and
    a Mx1 vector x.

    Parameters
    ----------
    A : array 2D
        NxM matrix.

    x : array 1D
        Mx1 vector.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Defines the matvec_real function to be used. The possible
        values are 'simple', 'dot' and 'columns'.

    Returns
    -------
    result : array 1D
        Product of A and x.
        Linear sistem = y = Ax
    '''
    x = np.asarray(x, dtype=complex)
    A = np.asarray(A, dtype=complex)

    if check_input:
        # Verifica se function é string
        if not isinstance(function, str):
            raise TypeError("function parameter must be a string")

        if function  not in ['simple', 'dot', 'columns']:
            raise ValueError(f"function must be one of; simple, dot, columns! Got {function}.")

    else:
        function = 'dot'

    matvec_real = {
        'simple' : matvec_real_simple,
        'dot' : matvec_real_dot,
        'columns' : matvec_real_columns
    }


    # use the syntax matvec_real[function] to specify the
    C_real = matvec_real[function](A.real, x.real)
    C_real -= matvec_real[function](A.imag, x.imag)
    C_imag = matvec_real[function](A.real, x.imag)
    C_imag += matvec_real[function](A.imag, x.real)
    # the matvec_real_* function.

    result = C_real + 1j*C_imag
    return result

# matrix-matrix product

def matmat_real_simple(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxL and B in R^LxM. The imaginary parts are ignored.

    The code uses a simple triply nested "for" to iterate on the arrays.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''

    # With:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))
    for i in range(0, N):
        for j in range(0,M):
            for k in range(0, L):
                result[i, j] += A[i, k]*B[k, j]
    
    return result


def matmat_real_dot(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces one "for" by a dot product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))

    for i in range(0, N):
        for j in range(0, M):
            result[i,:] = dot_real(A[i, :], B[:, j])

    return result


def matmat_real_rows(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "fors" by a matrix-vector product defining
    a row of the resultant matrix.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))

    for i in range(0, N):
        result[i, :] = matvec_real_dot(B[:, :].T , A[i, :])

    return result


def matmat_real_columns(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "fors" by a matrix-vector product defining
    a column of the resultant matrix.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))

    for j in range(0, M):
        result[:, j] = matvec_real_dot(A[:,:], B[:, j])

    return result


def matmat_real_outer(A, B, check_input=True):
    '''
    Compute the matrix-matrix product of A and B, where
    A in R^NxM and B in R^MxP. The imaginary parts are ignored.

    The code replaces two "fors" by an outer product.

    Parameters
    ----------
    A, B : 2D arrays
        Real matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        B = np.asanyarray(B, dtype=float)

    N, L = A.shape
    P, M = B.shape

    if check_input:
        # Testar se numeros de colunas em A{L} é igual ao numeros de linha em B{P}!
        assert L == P, ValueError(f"Columns of A ({L}) must match rows of B ({P}) for matrix multiplication")
        assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), TypeError("Inputs must be NumPy arrays")
        assert A.ndim  == 2 and B.ndim==2, "Both A and B must be 2D arrays."
    result = np.zeros(shape=(N, M))

    for  k in range(0, L):
        result[:,:] += outer_real_row(A[:,k], B[k,:])

    return result


def matmat_complex(A, B, check_input=True, function='simple'):
    '''
    Compute the matrix-matrix product of A and B, where
    A in C^NxM and B in C^MxP.

    Parameters
    ----------
    A, B : 2D arrays
        Complex matrices.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    function : string
        Defines the matmat_real function to be used. The possible
        values are 'simple', 'dot', 'rows', 'columns' or 'outer'.

    Returns
    -------
    result : 2D array
        Product of A and B.
    '''

    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)

    if check_input:
        # Verifica se function é string
        if not isinstance(function, str):
            raise TypeError("function parameter must be a string")

        if function  not in ['simple', 'dot', 'rows', 'columns', 'outer']:
            raise ValueError(f"function must be one of; simple, dot, rows, columns, outer! Got {function}.")

    else:
        function = 'simple'

    matmat_real = {
        'simple' : matmat_real_simple,
        'dot' : matmat_real_dot,
        'rows' : matmat_real_rows,
        'columns' : matmat_real_columns,
        'outer' : matmat_real_outer
    }

    # use the syntax matmat_real[function] to specify the
    # the matmat_real_* function.
    C_real = matmat_real[function](A.real, B.real)
    C_real -= matmat_real[function](A.imag, B.imag)
    C_imag = matmat_real[function](A.real, B.imag)
    C_imag += matmat_real[function](A.imag, B.real)
    result = C_real +1j*C_imag

    return result

## Triangular matrices
def matvec_triu_prod3(U, x, check_input=True):
    '''
    Compute the product of an upper triangular matrix U 
    and a vector x. All elements are real numbers.
    
    Each element of the resultant vector is obtained by 
    computing a dot product.

    Parameters
    ----------
    U : numpy array 2d
        Upper triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''

    # create your code here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        U = np.asanyarray(U, dtype=float)
        x = np.asanyarray(x, dtype=float)

    N, M = U.shape
    L = x.shape[0]

    if check_input:
        assert N==M, "U must be squere."
        assert isinstance(U, np.ndarray) and U.ndim == 2, "U must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == L, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(U, np.triu(U)), "U must be upper triangular" 
    result = np.zeros(L)

    for i in range(0, N):
        result[i] = dot_real(U[i, i:], x[i:])
    
    return result

def matvec_triu_prod5(U, x, check_input=True):
    '''
    Compute the product of an upper triangular matrix U 
    and a vector x. All elements are real numbers.
    
    The elements of the resultant vector are obtained by 
    computing successive scalar vector products.

    Parameters
    ----------
    U : numpy array 2d
        Upper triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''

    # create your code here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        U = np.asanyarray(U, dtype=float)
        x = np.asanyarray(x, dtype=float)

    N, M = U.shape
    L = x.shape[0]

    if check_input:
        assert N==M, "U must be squere."
        assert isinstance(U, np.ndarray) and U.ndim == 2, "U must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == L, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(U, np.triu(U)), "U must be upper triangular" 
    result = np.zeros_like(x)
    # obs, ao usar :j é preciso somar 1 para acessar o ultimo elemnto
    #   pois o laço é de N-1, e indexado no inicio em 0
    for j in range(0, N):
        result[:j+1] += scalar_vec_real(a=x[j], x=U[:j+1,j])
    
    return result

def matvec_tril_prod8(L, x, check_input=True):
    '''
    Compute the product of an lower triangular matrix L 
    and a vector x. All elements are real numbers.
    
    Each element of the resultant vector is obtained by 
    computing a dot product.

    Parameters
    ----------
    L : numpy array 2d
        Lower triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''

    # create your code here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        L = np.asanyarray(L, dtype=float)
        x = np.asanyarray(x, dtype=float)

    N, M = L.shape
    P = x.shape[0]

    if check_input:
        assert N==M, "L must be squere"
        assert isinstance(L, np.ndarray) and L.ndim == 2, "L must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == P, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(L, np.tril(L)), "L must be lower triangular" 
    result = np.zeros(P)
    # obs, ao usar :j é preciso somar 1 para acessar o ultimo elemnto
    #   pois o laço é de N-1, e indexado no inicio em 0
    for i in range(0, N):
        result[i] = dot_real(L[i,:i+1], x[:i+1])
    
    return result

def matvec_tril_prod10(L, x, check_input=True):
    '''
    Compute the product of an lower triangular matrix L 
    and a vector x. All elements are real numbers.
    
    The elements of the resultant vector are obtained by 
    computing successive scalar vector products.

    Parameters
    ----------
    L : numpy array 2d
        Lower triangular matrix.
    x : numpy array 1d
        Vector that postmultiply the triangular matrix U.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Vector obtained from the product U x.
    '''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        L = np.asanyarray(L, dtype=float)
        x = np.asanyarray(x, dtype=float)

    N, M = L.shape
    P = x.shape[0]

    if check_input:
        assert N==M, "L must be squere."
        assert isinstance(L, np.ndarray) and L.ndim == 2, "L must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == P, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(L, np.tril(L)), "L must be lower triangular" 
    
    result = np.zeros(P)
    # create your code here
    for j in range(0, N):
        result[j:] += scalar_vec_real(a=x[j], x=L[j:, j])
    return result
 

def triu_system(A, x, check_input=True):
    '''
    Solve the linear system Ax = y for x by using back substitution.

    The elements of x are computed by using a 'dot' within a single for.

    Parameters
    ----------
    A : numpy array 2d
        Upper triangular matrix.
    y : numpy array 1d
        Independent vector of the linear system.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Solution x of the linear system.
        Ay = x
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        x = np.asanyarray(x, dtype=float)
    N, M = A.shape
    n = x.shape[0]

    if check_input:
        assert N==M, "U must be squere."
        assert isinstance(A, np.ndarray) and A.ndim == 2, "U must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == n, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(A, np.triu(A)), "U must be upper triangular" 

    # create your code here

    result = np.zeros(n)

    for i in range(N-1, -1, -1):
        result[i] = x[i]  - dot_real(A[i, i+1:], result[i+1:])
        result[i] /= A[i, i]
    
    return result



def tril_system(A, x, check_input=True):
    '''
    Solve the linear system Ax = y for x by using forward substitution.

    The elements of x are computed by using a 'dot' within a single for.

    Parameters
    ----------
    A : numpy array 2d
        Lower triangular matrix.
    y : numpy array 1d
        Independent vector of the linear system.
    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : numpy array 1d
        Solution x of the linear system.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        A = np.asanyarray(A, dtype=float)
        x = np.asanyarray(x, dtype=float)
    N, M = A.shape
    n = x.shape[0]

    if check_input:
        assert N==M, "U must be squere."
        assert isinstance(A, np.ndarray) and A.ndim == 2, "U must be a 2D numpy array"
        assert isinstance(x, np.ndarray) and x.ndim == 1, "x must be a 1D numpy array"
        assert M == n, "Matrix and vector dimensions must match"
        # Verify U is actually upper triangular
        assert np.allclose(A, np.tril(A)), "L must be lower triangular" 

    # create your code here
    result = np.zeros(n)

    for i in range(0, N):
        result[i] = x[i]
        for j in range(0, i):
            result[i] -= A[i, j]*result[j]
        result[i]/=A[i,i]
    
    return result
