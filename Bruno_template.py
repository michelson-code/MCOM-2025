import numpy as np
from scipy.linalg import dft

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
        assert type(x) == np.ndarray, 'x must be a numpy array'
        assert x.ndim == 1, 'x must have ndim = 1'

    result = np.empty_like(x)
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
    def check(x, y):
        assert len(x) == len(y), 'Numero de elementos em x é diferente de numero de elementos em y'
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        # Testando se os valores são números reais (não pode ser complexo ou str)
        assert all(not isinstance(value, (str)) and 
                   not isinstance(value, complex) for value in x), "O array x contém valores que não são números reais"
        assert all(not isinstance(value, (str)) and 
                   not isinstance(value, complex) for value in y), "O array y contém valores que não são números reais"

    if check_input == True:
        check(x, y)
    elif check_input == False:
        pass
    
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
    if check_input==True:
        assert np.size(x) == np.size(y), "Número de elementos de x diferente de y!"
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
    else:
        pass
    assert all(isinstance(val, (complex, int, float)) for val in x), "Vetor x não pertence ao cunjunto dos complexos."
    assert all(not isinstance(val, str) for val in x), "Não deve conter string."
    assert all(isinstance(val, (complex, int, float)) for val in y), "Vetor y não pertence ao cunjunto dos complexos."
    # Extraindo partes reais e imaginárias
    
    x_real = np.real(x)
    x_imag = np.imag(x)
    y_real = np.real(y)
    y_imag = np.imag(y)
    
    #calculando o pruto escalar
    c1_real = dot_real(x_real, y_real) 
    c2_real = dot_real(x_imag, y_imag) #termo que subtrai do real
    c1_imag = dot_real(x_real, y_imag)
    c2_imag = dot_real(x_imag, y_real) #termo que soma do imaginario
    result = (c1_real - c2_real) + 1j*(c1_imag + c2_imag)
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
    N = np.size(x)
    M = np.size(y)
    result = np.zeros((N, M))
    x = np.asarray(x).real # Garante que x seja um array NumPy e pega apenas a parte real
    y = np.asarray(y).real # Garante que y seja um array NumPy e pega apenas a parte real
    

    if check_input == True:

        assert not(x.dtype == complex and y.dtype == complex), TypeError('Somente valores reais. Caso queira operar com valores complexo, utilize a função outer complex!')
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert np.isrealobj(x), 'x deve conter apenas números reais'
        assert np.isrealobj(y), 'y deve conter apenas números reais'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'

    for i in range(0, N):
        for j in range(0, M):
            result[i, j] = x[i]*y[j]

    return result.real


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
    N = np.size(x)
    M = np.size(y)
    result = np.zeros((N, M))
    x = np.asarray(x).real # Garante que x seja um array NumPy e pega apenas a parte real
    y = np.asarray(y).real # Garante que y seja um array NumPy e pega apenas a parte real

    if check_input == True:
        assert not(x.dtype == complex and y.dtype == complex), TypeError('Somente valores reais. Caso queira operar com valores complexo, utilize a função outer complex!')
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert np.isrealobj(x), 'x deve conter apenas números reais'
        assert np.isrealobj(y), 'y deve conter apenas números reais'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'

    for i in range(0, N):
        result[i][:] = x[i]*y[:]

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
    N = np.size(x)
    M = np.size(y)
    result = np.zeros((N, M))
    x = np.asarray(x).real # Garante que x seja um array NumPy e pega apenas a parte real
    y = np.asarray(y).real # Garante que y seja um array NumPy e pega apenas a parte real

    if check_input == True:
        assert not(x.dtype == complex and y.dtype == complex), TypeError('Somente valores reais. Caso queira operar com valores complexo, utilize a função outer complex!')
        assert isinstance(x, np.ndarray), 'x deve ser um numpy array, ex: numpy.array([])'
        assert isinstance(y, np.ndarray), 'y deve ser um numpy array, ex: numpy.array([])'
        assert np.isrealobj(x), 'x deve conter apenas números reais'
        assert np.isrealobj(y), 'y deve conter apenas números reais'
        assert x.ndim == 1, 'x deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'
        assert y.ndim == 1, 'y deve ser 1D com ndim = 1: uma dimensão [1, 2, 3]..'


    for j in range(M):
        result[:, j] = x[:] * y[j]
    return result


def outer_complex(x, y, check_input=True):
    '''
    Compute the outer product of x and y, where x and y are complex vectors.

    Parameters
    ----------
    x, y : 1D arrays
        Complex vectors.

    check_input : boolean
        If True, verify if the input is valid. Default is True.

    Returns
    -------
    result : 2D array
        Outer product of x and y.
    '''
    N = np.size(x)
    M = np.size(y)
    
    if check_input == True:
        # Verifica se x e y são vetores 1D
        assert isinstance(x, np.ndarray) and x.ndim == 1, 'x deve ser um vetor 1D'
        assert isinstance(y, np.ndarray) and y.ndim == 1, 'y deve ser um vetor 1D'
        # Verifica se x e y contêm números complexos
        assert np.iscomplexobj(x), 'x deve conter números complexos'
        assert np.iscomplexobj(y), 'y deve conter números complexos'
    else:
        pass
    # Calcula as partes reais e imaginárias do produto externo
    M_R = outer_real_simple(np.real(x), np.real(y))
    M_R -= outer_real_simple(np.imag(x), np.imag(y))
    M_I = outer_real_simple(np.real(x), np.imag(y))
    M_I += outer_real_simple(np.imag(x), np.real(y))
    # Combina as partes reais e imaginárias
    result = M_R + 1j * M_I    
    
    return result
