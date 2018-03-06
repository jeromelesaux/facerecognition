package algorithm

import (
	"math"
)

/** LU Decomposition.
<P>
For an m-by-n matrix A with m >= n, the LU decomposition is an m-by-n
unit lower triangular matrix L, an n-by-n upper triangular matrix U,
and a permutation vector piv of length m so that A(piv,:) = L*U.
If m < n, then L is m-by-m and U is m-by-n.
<P>
The LU decompostion with pivoting always exists, even if the matrix is
singular, so the constructor will never fail.  The primary use of the
LU decomposition is in the solution of square systems of simultaneous
linear equations.  This will fail if isNonsingular() returns false.
*/

type LUDecomposition struct {
	LU [][]float64 // Array for internal storage of decomposition.
	N  int         /** Row and column dimensions, and pivot sign.
	  column dimension.
	  l row dimension.
	  pivot sign. */
	M       int
	Pivsign int
	Piv     []int //Internal storage of pivot vector.
}

/** LU Decomposition
    Structure to access L, U and piv.
@param  A Rectangular matrix
*/

func NewLUDecomposition(a *Matrix) *LUDecomposition {
	lud := &LUDecomposition{}

	// Use a "left-looking", dot-product, Crout/Doolittle algorithm.

	lud.M = a.RowsDimension()
	lud.N = a.ColumnsDimension()
	lud.Piv = make([]int, lud.M)
	lud.LU = make([][]float64, a.M)

	for i := 0; i < a.M; i++ {
		lud.LU[i] = make([]float64, a.N)
	}
	for i := 0; i < a.M; i++ {
		for j := 0; j < a.N; j++ {
			lud.LU[i][j] = a.A[i][j]
		}
	}

	for i := 0; i < lud.M; i++ {
		lud.Piv[i] = i
	}
	pivsign := 1

	LUcolj := make([]float64, lud.M)

	// Outer loop.

	for j := 0; j < lud.N; j++ {

		// Make a copy of the j-th column to localize references.

		for i := 0; i < lud.M; i++ {
			LUcolj[i] = lud.LU[i][j]
		}

		// Apply previous transformations.

		for i := 0; i < lud.M; i++ {
			LUrowi := lud.LU[i]

			// Most of the time is spent in the following dot product.

			kmax := minInt(i, j)
			s := 0.0
			for k := 0; k < kmax; k++ {
				s += LUrowi[k] * LUcolj[k]
			}

			LUcolj[i] -= s
			LUrowi[j] = LUcolj[i]
		}

		// Find pivot and exchange if necessary.
		p := j
		for i := j + 1; i < lud.M; i++ {
			if math.Abs(LUcolj[i]) > math.Abs(LUcolj[p]) {
				p = i
			}
		}
		if p != j {
			for k := 0; k < lud.N; k++ {
				t := lud.LU[p][k]
				lud.LU[p][k] = lud.LU[j][k]
				lud.LU[j][k] = t
			}
			k := lud.Piv[p]
			lud.Piv[p] = lud.Piv[j]
			lud.Piv[j] = k
			pivsign = -pivsign
		}

		// Compute multipliers.

		if j < lud.M && lud.LU[j][j] != 0.0 {
			for i := j + 1; i < lud.M; i++ {
				lud.LU[i][j] /= lud.LU[j][j]
			}
		}
	}
	return lud
}

/* ------------------------
   Public Methods
 * ------------------------ */

/** Is the matrix nonsingular?
@return     true if U, and hence A, is nonsingular.
*/

func (lud *LUDecomposition) IsNonsingular() bool {
	for j := 0; j < lud.N; j++ {
		if lud.LU[j][j] == 0 {
			return false
		}
	}
	return true
}

/** Return lower triangular factor
@return     L
*/
func (lud *LUDecomposition) GetL() *Matrix {

	x := NewMatrix(lud.M, lud.N)

	for i := 0; i < lud.M; i++ {
		for j := 0; j < lud.N; j++ {
			if i > j {
				x.A[i][j] = lud.LU[i][j]
			} else {
				if i == j {
					x.A[i][j] = 1.0
				} else {
					x.A[i][j] = 0.0
				}
			}
		}
	}
	return x
}

/** Return upper triangular factor
@return     U
*/
func (lud *LUDecomposition) GetU() *Matrix {
	x := NewMatrix(lud.M, lud.N)

	for i := 0; i < lud.N; i++ {
		for j := 0; j < lud.N; j++ {
			if i <= j {
				x.A[i][j] = lud.LU[i][j]
			} else {
				x.A[i][j] = 0.0
			}
		}
	}
	return x
}

/** Return pivot permutation vector
@return     piv
*/

func (lud *LUDecomposition) GetPivot() []int {
	p := make([]int, lud.M)
	for i := 0; i < lud.M; i++ {
		p[i] = lud.Piv[i]
	}
	return p
}

/** Return pivot permutation vector as a one-dimensional double array
@return     (double) piv
*/
func (lud *LUDecomposition) GetDoublePivot() []float64 {
	vals := make([]float64, lud.M)
	for i := 0; i < lud.M; i++ {
		vals[i] = float64(lud.Piv[i])
	}
	return vals
}

/** Determinant
@return     det(A)
*/
func (lud *LUDecomposition) Det() float64 {
	if lud.M != lud.N {
		return -1.0
	}
	d := float64(lud.Pivsign)
	for j := 0; j < lud.N; j++ {
		d *= lud.LU[j][j]
	}
	return d
}

/** Solve A*X = B
@param  B   A Matrix with as many rows as A and any number of columns.
@return     X so that L*U*X = B(piv,:)
*/

func (lud *LUDecomposition) Solve(b *Matrix) *Matrix {

	if b.RowsDimension() != lud.M {
		return &Matrix{}
	}
	if !lud.IsNonsingular() {
		return &Matrix{}
	}

	// Copy right hand side with pivoting
	nx := b.ColumnsDimension()
	xmat := b.GetMatrix(lud.Piv, 0, nx-1)

	// Solve L*Y = B(piv,:)
	for k := 0; k < lud.N; k++ {
		for i := k + 1; i < lud.N; i++ {
			for j := 0; j < nx; j++ {
				xmat.A[i][j] -= xmat.A[k][j] * lud.LU[i][k]
			}
		}
	}
	// Solve U*X = Y;
	for k := lud.N - 1; k >= 0; k-- {
		for j := 0; j < nx; j++ {
			xmat.A[k][j] /= lud.LU[k][k]
		}
		for i := 0; i < k; i++ {
			for j := 0; j < nx; j++ {
				xmat.A[i][j] -= xmat.A[k][j] * lud.LU[i][k]
			}
		}
	}
	return xmat
}
