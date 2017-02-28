package algorithm

import (
	"facerecognition/logger"
	"github.com/pkg/errors"
)

/**
   Jama = Java Matrix class.
<P>
   The Java Matrix Class provides the fundamental operations of numerical
   linear algebra.  Various constructors create Matrices from two dimensional
   arrays of double precision floating point numbers.  Various "gets" and
   "sets" provide access to submatrices and matrix elements.  Several methods
   implement basic matrix arithmetic, including matrix addition and
   multiplication, matrix norms, and element-by-element array operations.
   Methods for reading and printing matrices are also included.  All the
   operations in this version of the Matrix Class involve real matrices.
   Complex matrices may be handled in a future version.
<P>
   Five fundamental matrix decompositions, which consist of pairs or triples
   of matrices, permutation vectors, and the like, produce results in five
   decomposition classes.  These decompositions are accessed by the Matrix
   class to compute solutions of simultaneous linear equations, determinants,
   inverses and other matrix functions.  The five decompositions are:
<P><UL>
   <LI>Cholesky Decomposition of symmetric, positive definite matrices.
   <LI>LU Decomposition of rectangular matrices.
   <LI>QR Decomposition of rectangular matrices.
   <LI>Singular Value Decomposition of rectangular matrices.
   <LI>Eigenvalue Decomposition of both symmetric and nonsymmetric square matrices.
</UL>
<DL>
<DT><B>Example of use:</B></DT>
<P>
<DD>Solve a linear system A x = b and compute the residual norm, ||b - A x||.
<P><PRE>
      double[][] vals = {{1.,2.,3},{4.,5.,6.},{7.,8.,10.}};
      Matrix A = new Matrix(vals);
      Matrix b = Matrix.random(3,1);
      Matrix x = A.solve(b);
      Matrix r = A.times(x).minus(b);
      double rnorm = r.normInf();
</PRE></DD>
</DL>

@author The MathWorks, Inc. and the National Institute of Standards and Technology.
@version 5 August 1998
*/
type Matrix struct {
	A [][]float64
	M int
	N int
}

func NewMatrix(m, n int) *Matrix {
	mat := &Matrix{
		M: m,
		N: n,
	}
	a := make([][]float64, m)
	for i := 0; i < m; i++ {
		a[i] = make([]float64, n)
	}
	mat.A = a
	return mat
}

func NewMatrixFilled(m, n, s int) *Matrix {
	mat := NewMatrix(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			mat.A[i][j] = float64(s)
		}
	}
	return mat
}
func NewMatrixWithArrays(a [][]float64) (*Matrix, error) {
	m := len(a)
	n := len(a[0])
	mat := NewMatrix(m, n)
	for i := 0; i < m; i++ {
		if len(a[i]) != n {
			return nil, errors.New("All rows must have the same length.")
		}
		for j := 0; j < n; j++ {
			mat.A[i][j] = a[i][j]
		}
	}
	return mat, nil
}

func NewMatrixWithMatrix(a [][]float64, m, n int) (*Matrix, error) {
	return NewMatrixWithArrays(a)
}

func (m *Matrix) Transpose() *Matrix {
	x := NewMatrix(m.M, m.N)
	for i := 0; i < m.M; i++ {
		for j := 0; j < m.N; j++ {
			x.A[j][i] = m.A[i][j]
		}
	}
	return x
}

func (m *Matrix) Plus(b *Matrix) *Matrix {
	x := NewMatrix(m.M, m.N)
	for i := 0; i < m.M; i++ {
		for j := 0; j < m.N; j++ {
			x.A[i][j] = m.A[i][j] + b.A[i][j]
		}
	}
	return x
}

func (m *Matrix) PlusEqual(b *Matrix) {
	for i := 0; i < m.M; i++ {
		for j := 0; j < m.N; j++ {
			m.A[i][j] = m.A[i][j] + b.A[i][j]
		}
	}
	return
}

func (m *Matrix) Minus(b *Matrix) *Matrix {
	x := NewMatrix(m.M, m.N)
	for i := 0; i < m.M; i++ {
		for j := 0; j < m.N; j++ {
			x.A[i][j] = m.A[i][j] - b.A[i][j]
		}
	}
	return x
}

func (m *Matrix) TimesMatrix(b *Matrix) *Matrix {
	if b.M != b.N {
		logger.Log("Matrix inner dimensions must agree")
		return &Matrix{}
	}
	x := NewMatrix(m.M, b.N)
	Bcolj := make([]float64, m.N)
	for j := 0; j < b.N; j++ {
		for k := 0; k < m.N; k++ {
			Bcolj[k] = b.A[k][j]
		}
		for i := 0; i < m.M; i++ {
			Arowi := m.A[i]
			s := 0.0
			for k := 0; k < m.N; k++ {
				s += Arowi[k] * Bcolj[k]
			}
			x.A[i][j] = s
		}
	}
	return x
}

func (m *Matrix) Times(s float64) *Matrix {
	x := NewMatrix(m.M, m.N)
	for i := 0; i < m.M; i++ {
		for j := 0; j < m.N; j++ {
			x.A[i][j] = s * m.A[i][j]
		}
	}
	return x
}

func (m *Matrix) RowsDimension() int {
	return m.M
}

func (m *Matrix) ColumnsDimension() int {
	return m.N
}

func (m *Matrix) GetMatrix(r []int, j0, j1 int) *Matrix {
	x := NewMatrix(len(r), j1-j0+1)
	for i := 0; i < len(r); i++ {
		for j := j0; j <= j1; j++ {
			x.A[i][j-j0] = m.A[r[i]][j]
		}
	}
	return x
}

func (m *Matrix) GetMatrix2(i0, i1 int, c []int) *Matrix {
	x := NewMatrix(i1-i0+1, len(c))
	for i := i0; i <= i1; i++ {
		for j := 0; j < len(c); j++ {
			x.A[i-i0][j] = m.A[i][c[j]]
		}
	}
	return x
}

func (m *Matrix) SetMatrix(i0, i1, j0, j1 int, x *Matrix) {
	for i := i0; i <= i1; i++ {
		for j := j0; j <= j1; j++ {
			m.A[i][j] = x.A[i-i0][j-j0]
		}
	}
}

func (m *Matrix) GetMatrix3(i0, i1, j0, j1 int) *Matrix {
	x := NewMatrix(i1-i0+1, j1-j0+1)
	for i := i0; i <= i1; i++ {
		for j := j0; j <= j1; j++ {
			x.A[i-i0][j-j0] = m.A[i][j]
		}
	}
	return x
}

func (m *Matrix) Eig() *EigenvalueDecomposition {
	return NewEigenvalueDecomposition(m)
}

func (m *Matrix) Inverse() *Matrix {
	return m.Solve(m.Identity(m.M, m.M))
}

func (m *Matrix) Solve(b *Matrix) *Matrix {
	if m.M == m.N {
		return NewLUDecomposition(m).Solve(b)
	} else {
		return NewQRDecomposition(m).Solve(b)
	}
}

func (mat *Matrix) Identity(m, n int) *Matrix {
	a := NewMatrix(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				a.A[i][j] = 1.0
			} else {
				a.A[i][j] = .0
			}
		}
	}
	return a
}
