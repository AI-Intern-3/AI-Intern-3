## 1. NumPy

# Advantages:
Fundamental package for scientific computing with Python.
Provides support for large, multi-dimensional arrays and matrices.
Includes a collection of mathematical functions to operate on these arrays.

# Use Cases:
Numerical computations.
Linear algebra operations.
Statistical analysis.
Fourier transforms.

NumPy functions categorized by their use in array operations, linear algebra, Fourier transforms, and statistical analysis, along with brief descriptions of their uses:

## Array Operations
- **Array Creation:**
  - `np.array()`: Create an array from a list or tuple.
  - `np.zeros()`: Create an array filled with zeros. Useful for initializing arrays.
  - `np.ones()`: Create an array filled with ones. Useful for initializing arrays.
  - `np.empty()`: Create an uninitialized array. Useful when the initial content does not matter.
  - `np.full()`: Create an array filled with a specified value.
  - `np.arange()`: Create an array with evenly spaced values within a given interval.
  - `np.linspace()`: Create an array with evenly spaced values over a specified range.
  - `np.logspace()`: Create an array with logarithmically spaced values.
  - `np.eye()`: Create a 2-D array with ones on the diagonal and zeros elsewhere. Useful for identity matrices.
  - `np.identity()`: Create an identity matrix.

- **Array Manipulation:**
  - `np.reshape()`: Change the shape of an array without changing its data.
  - `np.ravel()`: Return a contiguous flattened array.
  - `np.flatten()`: Return a copy of the array collapsed into one dimension.
  - `np.transpose()`: Permute the dimensions of an array.
  - `np.swapaxes()`: Interchange two axes of an array.
  - `np.concatenate()`: Join a sequence of arrays along an existing axis.
  - `np.hstack()`: Stack arrays in sequence horizontally (column-wise).
  - `np.vstack()`: Stack arrays in sequence vertically (row-wise).
  - `np.split()`: Split an array into multiple sub-arrays.
  - `np.hsplit()`: Split an array into multiple sub-arrays horizontally (column-wise).
  - `np.vsplit()`: Split an array into multiple sub-arrays vertically (row-wise).

- **Mathematical Operations:**
  - `np.add()`: Element-wise addition of arrays.
  - `np.subtract()`: Element-wise subtraction of arrays.
  - `np.multiply()`: Element-wise multiplication of arrays.
  - `np.divide()`: Element-wise division of arrays.
  - `np.power()`: First array elements raised to the powers from the second array, element-wise.
  - `np.mod()`: Element-wise remainder of division.
  - `np.sqrt()`: Return the non-negative square root of an array, element-wise.
  - `np.exp()`: Calculate the exponential of all elements in the input array.
  - `np.log()`: Natural logarithm, element-wise.
  - `np.log10()`: Return the base 10 logarithm of the input array, element-wise.
  - `np.sin()`, `np.cos()`, `np.tan()`: Trigonometric functions, element-wise.
  - `np.arcsin()`, `np.arccos()`, `np.arctan()`: Inverse trigonometric functions, element-wise.

### Linear Algebra
- **Basic Linear Algebra Operations:**
  - `np.dot()`: Dot product of two arrays.
  - `np.vdot()`: Dot product of two vectors.
  - `np.matmul()`: Matrix product of two arrays.
  - `np.inner()`: Inner product of two arrays.
  - `np.outer()`: Outer product of two arrays.

- **Matrix Decomposition:**
  - `np.linalg.det()`: Compute the determinant of an array.
  - `np.linalg.inv()`: Compute the (multiplicative) inverse of a matrix.
  - `np.linalg.eig()`: Compute the eigenvalues and right eigenvectors of a square array.
  - `np.linalg.svd()`: Singular Value Decomposition.
  - `np.linalg.qr()`: Compute the QR decomposition of a matrix.

- **Solving Equations and Inverting Matrices:**
  - `np.linalg.solve()`: Solve a linear matrix equation, or system of linear scalar equations.
  - `np.linalg.lstsq()`: Compute the least-squares solution to a linear matrix equation.
  - `np.linalg.pinv()`: Compute the (Moore-Penrose) pseudo-inverse of a matrix.

- **Norms and Other Numbers:**
  - `np.linalg.norm()`: Compute the matrix or vector norm.
  - `np.trace()`: Return the sum along diagonals of the array.

### Fourier Transforms
- **Discrete Fourier Transform (DFT):**
  - `np.fft.fft()`: Compute the one-dimensional discrete Fourier Transform.
  - `np.fft.ifft()`: Compute the one-dimensional inverse discrete Fourier Transform.
  - `np.fft.fft2()`: Compute the two-dimensional discrete Fourier Transform.
  - `np.fft.ifft2()`: Compute the two-dimensional inverse discrete Fourier Transform.
  - `np.fft.fftn()`: Compute the N-dimensional discrete Fourier Transform.
  - `np.fft.ifftn()`: Compute the N-dimensional inverse discrete Fourier Transform.

- **Discrete Cosine and Sine Transforms:**
  - `np.fft.dct()`: Compute the discrete cosine transform.
  - `np.fft.idct()`: Compute the inverse discrete cosine transform.
  - `np.fft.dst()`: Compute the discrete sine transform.
  - `np.fft.idst()`: Compute the inverse discrete sine transform.

- **Frequency Analysis:**
  - `np.fft.fftfreq()`: Return the Discrete Fourier Transform sample frequencies.
  - `np.fft.fftshift()`: Shift the zero-frequency component to the center of the spectrum.
  - `np.fft.ifftshift()`: Inverse of `fftshift`.

### Statistical Analysis
- **Descriptive Statistics:**
  - `np.mean()`: Compute the arithmetic mean along the specified axis.
  - `np.median()`: Compute the median along the specified axis.
  - `np.var()`: Compute the variance along the specified axis.
  - `np.std()`: Compute the standard deviation along the specified axis.
  - `np.min()`, `np.max()`: Return the minimum and maximum of an array or minimum along an axis.
  - `np.percentile()`: Compute the nth percentile of the data along the specified axis.
  - `np.quantile()`: Compute the q-th quantile of the data along the specified axis.

- **Order Statistics:**
  - `np.sort()`: Return a sorted copy of an array.
  - `np.argsort()`: Indices that would sort an array.
  - `np.argmin()`, `np.argmax()`: Indices of the minimum and maximum values along an axis.

- **Correlation and Covariance:**
  - `np.corrcoef()`: Return Pearson product-moment correlation coefficients.
  - `np.cov()`: Estimate a covariance matrix.

- **Histograms and Binning:**
  - `np.histogram()`: Compute the histogram of a dataset.
  - `np.histogram2d()`: Compute the bi-dimensional histogram of two data samples.
  - `np.histogramdd()`: Compute the multidimensional histogram of some data.
  - `np.bincount()`: Count the number of occurrences of each value in an array of non-negative ints.

- **Random Sampling:**
  - `np.random.rand()`: Generate random numbers from a uniform distribution.
  - `np.random.randn()`: Generate random numbers from a standard normal distribution.
  - `np.random.randint()`: Generate random integers from a specified range.
  - `np.random.choice()`: Generate a random sample from a given 1-D array.
  - `np.random.seed()`: Seed the random number generator.

These functions cover a wide range of operations essential for numerical computations, linear algebra, Fourier transforms, and statistical analysis using NumPy.
## Numerical Computations

Numerical computations involve algorithms to solve mathematical problems approximately rather than analytically. Here are a few key concepts:

#### Example: Solving Equations Numerically
Suppose we want to find the roots of the equation \( f(x) = x^2 - 2 = 0 \). The exact solution is \( x = \sqrt{2} \approx 1.414 \).

One common numerical method is the **Newton-Raphson method**:
\[ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \]

1. Choose an initial guess, \( x_0 \).
2. Compute \( x_1 = x_0 - \frac{f(x_0)}{f'(x_0)} \).
3. Repeat until the desired accuracy is achieved.

#### Example Implementation in Python:
```python
def f(x):
    return x**2 - 2

def f_prime(x):
    return 2 * x

x_n = 1.0  # initial guess
tolerance = 1e-10

while abs(f(x_n)) > tolerance:
    x_n = x_n - f(x_n) / f_prime(x_n)

print(f"The root is approximately: {x_n}")
```

### Linear Algebra Operations

Linear algebra deals with vectors, matrices, and linear transformations.

#### Example: Solving a System of Linear Equations
Given a system of equations:
\[ 
\begin{cases}
2x + 3y = 5 \\
4x + y = 6 
\end{cases}
\]

We can represent this system as:
\[ AX = B \]
where \( A = \begin{pmatrix} 2 & 3 \\ 4 & 1 \end{pmatrix} \) and \( B = \begin{pmatrix} 5 \\ 6 \end{pmatrix} \).

To solve for \( X = \begin{pmatrix} x \\ y \end{pmatrix} \), we can use matrix operations:
\[ X = A^{-1}B \]

#### Example Implementation in Python:
```python
import numpy as np

A = np.array([[2, 3], [4, 1]])
B = np.array([5, 6])

X = np.linalg.solve(A, B)
print(f"The solution is x = {X[0]}, y = {X[1]}")
```

### Statistical Analysis

Statistical analysis involves collecting, analyzing, and interpreting data to draw conclusions.

#### Example: Descriptive Statistics
Given a dataset: \([1, 2, 3, 4, 5]\)

We can calculate:
- **Mean**: \(\mu = \frac{1+2+3+4+5}{5} = 3\)
- **Variance**: \(\sigma^2 = \frac{(1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2}{5} = 2\)
- **Standard Deviation**: \(\sigma = \sqrt{2} \approx 1.414\)

#### Example Implementation in Python:
```python
import numpy as np

data = [1, 2, 3, 4, 5]

mean = np.mean(data)
variance = np.var(data)
std_dev = np.std(data)

print(f"Mean: {mean}, Variance: {variance}, Standard Deviation: {std_dev}")
```

### Fourier Transforms

Fourier transforms decompose a function (signal) into its constituent frequencies.

#### Example: Fourier Transform of a Signal
Suppose we have a signal \( f(t) = \cos(2\pi t) + \cos(4\pi t) \).

The Fourier transform converts this time-domain signal into the frequency domain, showing the frequencies present in the signal.

#### Example Implementation in Python:
```python
import numpy as np
import matplotlib.pyplot as plt

# Time domain signal
t = np.linspace(0, 1, 400)
signal = np.cos(2 * np.pi * t) + np.cos(4 * np.pi * t)

# Compute Fourier Transform
frequency_domain = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(t), d=t[1] - t[0])

# Plot the signal
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Time Domain Signal")

# Plot the Fourier Transform (magnitude)
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(frequency_domain))
plt.title("Frequency Domain Signal")
plt.show()
```

Each of these examples demonstrates how these mathematical concepts can be implemented and applied using Python, making it easier to perform complex computations and analyses.
