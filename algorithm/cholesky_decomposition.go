package algorithm

import (
	"github.com/jeromelesaux/facerecognition/logger"
	"math"
)

/** Cholesky Decomposition.
  <P>
  For a symmetric, positive definite matrix A, the Cholesky decomposition
  is an lower triangular matrix L so that A = L*L'.
  <P>
  If the matrix is not symmetric or positive definite, the constructor
  returns a partial decomposition and sets an internal flag that may
  be queried by the isSPD() method.
*/
type CholeskyDecomposition struct {
	L     [][]float64 // Array for internal storage of decomposition.
	N     int         // Row and column dimension (square matrix).
	Isspd bool        // Symmetric and positive definite flag.
}

func NewCholeskyDecomposition(matrix *Matrix) *CholeskyDecomposition {
	c := &CholeskyDecomposition{N: matrix.RowsDimension(), Isspd: (matrix.ColumnsDimension() == matrix.RowsDimension())}
	c.L = make([][]float64, c.N)
	for i := 0; i < c.N; i++ {
		c.L[i] = make([]float64, c.N)
	}
	//double[][] A = Arg.getArray();

	// Main loop.
	for j := 0; j < c.N; j++ {
		Lrowj := c.L[j]
		d := 0.0
		for k := 0; k < j; k++ {
			Lrowk := c.L[k]
			s := 0.0
			for i := 0; i < k; i++ {
				s += Lrowk[i] * Lrowj[i]
			}
			Lrowj[k] = (matrix.A[j][k] - s) / c.L[k][k]
			s = Lrowj[k]
			d = d + s*s
			c.Isspd = c.Isspd && (matrix.A[k][j] == matrix.A[j][k])
		}
		d = matrix.A[j][j] - d
		c.Isspd = c.Isspd && (d > 0.0)
		c.L[j][j] = math.Sqrt(math.Max(d, 0.0))
		for k := j + 1; k < c.N; k++ {
			c.L[j][k] = 0.0
		}
	}
	return c
}

func (c *CholeskyDecomposition) GetL() *Matrix {
	mat, _ := NewMatrixWithMatrix(c.L, c.N, c.N)
	return mat
}

/** Solve A*X = B
@param  B   A Matrix with as many rows as A and any number of columns.
@return     X so that L*L'*X = B
@exception  IllegalArgumentException  Matrix row dimensions must agree.
@exception  RuntimeException  Matrix is not symmetric positive definite.
*/
func (c *CholeskyDecomposition) Solve(B *Matrix) *Matrix {
	if B.RowsDimension() != c.N {
		return &Matrix{}
	}
	if !c.Isspd {
		logger.Log("Matrix is not symmetric positive definite.")
		return &Matrix{}
	}

	var X [][]float64
	// Copy right hand side.
	copy(X, B.A)

	nx := B.ColumnsDimension()

	// Solve L*Y = B;
	for k := 0; k < c.N; k++ {
		for j := 0; j < nx; j++ {
			for i := 0; i < k; i++ {
				X[k][j] -= X[i][j] * c.L[k][i]
			}
			X[k][j] /= c.L[k][k]
		}
	}

	// Solve L'*X = Y;
	for k := c.N - 1; k >= 0; k-- {
		for j := 0; j < nx; j++ {
			for i := k + 1; i < c.N; i++ {
				X[k][j] -= X[i][j] * c.L[i][k]
			}
			X[k][j] /= c.L[k][k]
		}
	}

	mat, _ := NewMatrixWithMatrix(X, c.N, nx)
	return mat
}
