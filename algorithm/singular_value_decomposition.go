package algorithm

import (
	"math"
)

/** Singular Value Decomposition.
  <P>
  For an m-by-n matrix A with m >= n, the singular value decomposition is
  an m-by-n orthogonal matrix U, an n-by-n diagonal matrix S, and
  an n-by-n orthogonal matrix V so that A = U*S*V'.
  <P>
  The singular values, sigma[k] = S[k][k], are ordered so that
  sigma[0] >= sigma[1] >= ... >= sigma[n-1].
  <P>
  The singular value decompostion always exists, so the constructor will
  never fail.  The matrix condition number and the effective numerical
  rank can be computed from this decomposition.
*/

type SingularValueDecomposition struct {
	/* ------------------------
	   Class variables
	 * ------------------------ */

	/** Arrays for internal storage of U and V.
	@serial internal storage of U.
	@serial internal storage of V.
	*/
	U [][]float64
	V [][]float64
	/** Array for internal storage of singular values.
	@serial internal storage of singular values.
	*/
	S []float64

	/** Row and column dimensions.
	@serial row dimension.
	@serial column dimension.
	*/
	M int
	N int
}

/* ------------------------
   Constructor
 * ------------------------ */

/** Construct the singular value decomposition
    Structure to access U, S and V.
@param Arg    Rectangular matrix
*/

func NewSingularValueDecomposition(matrix *Matrix) *SingularValueDecomposition {

	s := &SingularValueDecomposition{M: matrix.RowsDimension(), N: matrix.ColumnsDimension()}
	// Derived from LINPACK code.
	// Initialize.

	//double[][] A = Arg.getArrayCopy();

	/* Apparently the failing cases are only a proper subset of (m<n),
	   so let's not throw error.  Correct fix to come later?
	if (m<n) {
	    throw new IllegalArgumentException("Jama SVD only works for m >= n"); }
	*/
	nu := minInt(s.M, s.N)
	s.S = make([]float64, minInt(s.M+1, s.N))
	s.U = make([][]float64, s.M)
	for i := 0; i < minInt(s.M+1, s.N); i++ {
		s.U[i] = make([]float64, nu)
	}
	s.V = make([][]float64, s.N)
	for i := 0; i < s.N; i++ {
		s.V[i] = make([]float64, s.N)
	}
	e := make([]float64, s.N)
	work := make([]float64, s.M)
	wantu := true
	wantv := true

	// Reduce A to bidiagonal form, storing the diagonal elements
	// in s and the super-diagonal elements in e.
	nct := minInt(s.M-1, s.N)
	nrt := maxInt(0, minInt(s.N-2, s.M))
	for k := 0; k < maxInt(nct, nrt); k++ {
		if k < nct {
			// Compute the transformation for the k-th column and
			// place the k-th diagonal in s[k].
			// Compute 2-norm of k-th column without under/overflow.
			s.S[k] = 0
			for i := k; i < s.M; i++ {
				s.S[k] = math.Hypot(s.S[k], matrix.A[i][k])
			}
			if s.S[k] != 0.0 {
				if matrix.A[k][k] < 0.0 {
					s.S[k] = -s.S[k]
				}
				for i := k; i < s.M; i++ {
					matrix.A[i][k] /= s.S[k]
				}
				matrix.A[k][k] += 1.0
			}
			s.S[k] = -s.S[k]
		}
		for j := k + 1; j < s.N; j++ {
			if (k < nct) && (s.S[k] != 0.0) {
				// Apply the transformation.
				t := 0.0
				for i := k; i < s.M; i++ {
					t += matrix.A[i][k] * matrix.A[i][j]
				}
				t = -t / matrix.A[k][k]
				for i := k; i < s.M; i++ {
					matrix.A[i][j] += t * matrix.A[i][k]
				}
			}
			// Place the k-th row of A into e for the
			// subsequent calculation of the row transformation.

			e[j] = matrix.A[k][j]
		}
		if wantu && (k < nct) {
			// Place the transformation in U for subsequent back
			// multiplication.
			for i := k; i < s.M; i++ {
				s.U[i][k] = matrix.A[i][k]
			}
		}
		if k < nrt {

			// Compute the k-th row transformation and place the
			// k-th super-diagonal in e[k].
			// Compute 2-norm without under/overflow.
			e[k] = 0
			for i := k + 1; i < s.N; i++ {
				e[k] = math.Hypot(e[k], e[i])
			}
			if e[k] != 0.0 {
				if e[k+1] < 0.0 {
					e[k] = -e[k]
				}
				for i := k + 1; i < s.N; i++ {
					e[i] /= e[k]
				}
				e[k+1] += 1.0
			}
			e[k] = -e[k]
			if (k+1 < s.M) && (e[k] != 0.0) {
				// Apply the transformation.
				for i := k + 1; i < s.M; i++ {
					work[i] = 0.0
				}
				for j := k + 1; j < s.N; j++ {
					for i := k + 1; i < s.M; i++ {
						work[i] += e[j] * matrix.A[i][j]
					}
				}
				for j := k + 1; j < s.N; j++ {
					t := -e[j] / e[k+1]
					for i := k + 1; i < s.M; i++ {
						matrix.A[i][j] += t * work[i]
					}
				}
			}
			if wantv {
				// Place the transformation in V for subsequent
				// back multiplication.
				for i := k + 1; i < s.N; i++ {
					s.V[i][k] = e[i]
				}
			}
		}
	}

	// Set up the final bidiagonal matrix or order p.
	p := minInt(s.N, s.M+1)
	if nct < s.N {
		s.S[nct] = matrix.A[nct][nct]
	}
	if s.M < p {
		s.S[p-1] = 0.0
	}
	if nrt+1 < p {
		e[nrt] = matrix.A[nrt][p-1]
	}
	e[p-1] = 0.0
	// If required, generate U.

	if wantu {
		for j := nct; j < nu; j++ {
			for i := 0; i < s.M; i++ {
				s.U[i][j] = 0.0
			}
			s.U[j][j] = 1.0
		}
		for k := nct - 1; k >= 0; k-- {
			if s.S[k] != 0.0 {
				for j := k + 1; j < nu; j++ {
					t := 0.0
					for i := k; i < s.M; i++ {
						t += s.U[i][k] * s.U[i][j]
					}
					t = -t / s.U[k][k]
					for i := k; i < s.M; i++ {
						s.U[i][j] += t * s.U[i][k]
					}
				}
				for i := k; i < s.M; i++ {
					s.U[i][k] = -s.U[i][k]
				}
				s.U[k][k] = 1.0 + s.U[k][k]
				for i := 0; i < k-1; i++ {
					s.U[i][k] = 0.0
				}
			} else {
				for i := 0; i < s.M; i++ {
					s.U[i][k] = 0.0
				}
				s.U[k][k] = 1.0
			}
		}
	}
	// If required, generate V.
	if wantv {
		for k := s.N - 1; k >= 0; k-- {
			if (k < nrt) && (e[k] != 0.0) {
				for j := k + 1; j < nu; j++ {
					t := 0.0
					for i := k + 1; i < s.N; i++ {
						t += s.V[i][k] * s.V[i][j]
					}
					t = -t / s.V[k+1][k]
					for i := k + 1; i < s.N; i++ {
						s.V[i][j] += t * s.V[i][k]
					}
				}
			}
			for i := 0; i < s.N; i++ {
				s.V[i][k] = 0.0
			}
			s.V[k][k] = 1.0
		}
	}
	// Main iteration loop for the singular values.
	pp := p - 1
	iter := 0
	eps := math.Pow(2.0, -52.0)
	tiny := math.Pow(2.0, -966.0)
	for p > 0 {
		k := 0
		kase := 0

		// Here is where a test for too many iterations would go.
		// This section of the program inspects for
		// negligible elements in the s and e arrays.  On
		// completion the variables kase and k are set as follows.

		// kase = 1     if s(p) and e[k-1] are negligible and k<p
		// kase = 2     if s(k) is negligible and k<p
		// kase = 3     if e[k-1] is negligible, k<p, and
		//              s(k), ..., s(p) are not negligible (qr step).
		// kase = 4     if e(p-1) is negligible (convergence).

		for k := p - 2; k >= -1; k-- {
			if k == -1 {
				break
			}
			if math.Abs(e[k]) <= tiny+eps*(math.Abs(s.S[k])+math.Abs(s.S[k+1])) {
				e[k] = 0.0
				break
			}
		}
		if k == p-2 {
			kase = 4
		} else {
			ks := 0
			for ks = p - 1; ks >= k; ks-- {
				if ks == k {
					break
				}
				t := 0.0
				if ks != p {
					if ks != k+1 {
						t = math.Abs(e[ks]) + math.Abs(e[ks-1])
					} else {
						t = math.Abs(e[ks])
					}
				} else {
					if ks != k+1 {
						t = math.Abs(e[ks-1])
					}
				}
				//double
				//t = (ks != p ? Math.abs(e[ks]) : 0.) + (ks != k + 1 ? Math.abs(e[ks - 1]) : 0.);
				if math.Abs(s.S[ks]) <= tiny+eps*t {
					s.S[ks] = 0.0
					break
				}
			}
			if ks == k {
				kase = 3
			} else if ks == p-1 {
				kase = 1
			} else {
				kase = 2
				k = ks
			}
		}
		k++

		// Perform the task indicated by kase.

		switch kase {

		// Deflate negligible s(p).
		case 1:
			{

				f := e[p-2]
				e[p-2] = 0.0
				for j := p - 2; j >= k; j-- {
					t := math.Hypot(s.S[j], f)
					cs := s.S[j] / t
					sn := f / t
					s.S[j] = t
					if j != k {
						f = -sn * e[j-1]
						e[j-1] = cs * e[j-1]
					}
					if wantv {
						for i := 0; i < s.N; i++ {
							t = cs*s.V[i][j] + sn*s.V[i][p-1]
							s.V[i][p-1] = -sn*s.V[i][j] + cs*s.V[i][p-1]
							s.V[i][j] = t
						}
					}
				}
			}
			break

		// Split at negligible s(k).

		case 2:
			{
				f := e[k-1]
				e[k-1] = 0.0
				for j := k; j < p; j++ {
					t := math.Hypot(s.S[j], f)
					cs := s.S[j] / t
					sn := f / t
					s.S[j] = t
					f = -sn * e[j]
					e[j] = cs * e[j]
					if wantu {
						for i := 0; i < s.M; i++ {
							t = cs*s.U[i][j] + sn*s.U[i][k-1]
							s.U[i][k-1] = -sn*s.U[i][j] + cs*s.U[i][k-1]
							s.U[i][j] = t
						}
					}
				}
			}
			break

		// Perform one qr step.

		case 3:
			{
				// Calculate the shift.
				scale := math.Max(math.Max(math.Max(math.Max(math.Abs(s.S[p-1]), math.Abs(s.S[p-2])), math.Abs(e[p-2])), math.Abs(s.S[k])), math.Abs(e[k]))
				sp := s.S[p-1] / scale
				spm1 := s.S[p-2] / scale
				epm1 := e[p-2] / scale
				sk := s.S[k] / scale
				ek := e[k] / scale
				b := ((spm1+sp)*(spm1-sp) + epm1*epm1) / 2.0
				c := (sp * epm1) * (sp * epm1)
				shift := 0.0
				if (b != 0.0) || (c != 0.0) {
					shift = math.Sqrt(b*b + c)
					if b < 0.0 {
						shift = -shift
					}
					shift = c / (b + shift)
				}

				f := (sk+sp)*(sk-sp) + shift
				g := sk * ek
				// Chase zeros.
				for j := k; j < p-1; j++ {
					t := math.Hypot(f, g)
					cs := f / t
					sn := g / t
					if j != k {
						e[j-1] = t
					}
					f = cs*s.S[j] + sn*e[j]
					e[j] = cs*e[j] - sn*s.S[j]
					g = sn * s.S[j+1]
					s.S[j+1] = cs * s.S[j+1]
					if wantv {
						for i := 0; i < s.N; i++ {
							t = cs*s.V[i][j] + sn*s.V[i][j+1]
							s.V[i][j+1] = -sn*s.V[i][j] + cs*s.V[i][j+1]
							s.V[i][j] = t
						}
					}
					t = math.Hypot(f, g)
					cs = f / t
					sn = g / t
					s.S[j] = t
					f = cs*e[j] + sn*s.S[j+1]
					s.S[j+1] = -sn*e[j] + cs*s.S[j+1]
					g = sn * e[j+1]
					e[j+1] = cs * e[j+1]
					if wantu && (j < s.M-1) {
						for i := 0; i < s.M; i++ {
							t = cs*s.U[i][j] + sn*s.U[i][j+1]
							s.U[i][j+1] = -sn*s.U[i][j] + cs*s.U[i][j+1]
							s.U[i][j] = t
						}
					}
				}
				e[p-2] = f
				iter = iter + 1
			}
			break
		// Convergence.
		case 4:
			{
				// Make the singular values positive.
				if s.S[k] <= 0.0 {
					if s.S[k] < 0.0 {
						s.S[k] = -s.S[k]
					} else {
						s.S[k] = 0.0
					}
					if wantv {
						for i := 0; i <= pp; i++ {
							s.V[i][k] = -s.V[i][k]
						}
					}
				}
				// Order the singular values.
				for k < pp {
					if s.S[k] >= s.S[k+1] {
						break
					}
					t := s.S[k]
					s.S[k] = s.S[k+1]
					s.S[k+1] = t
					if wantv && (k < s.N-1) {
						for i := 0; i < s.N; i++ {
							t = s.V[i][k+1]
							s.V[i][k+1] = s.V[i][k]
							s.V[i][k] = t
						}
					}
					if wantu && (k < s.M-1) {
						for i := 0; i < s.M; i++ {
							t = s.U[i][k+1]
							s.U[i][k+1] = s.U[i][k]
							s.U[i][k] = t
						}
					}
					k++
				}
				iter = 0
				p--
			}
			break
		}
	}
	return s
}

/* ------------------------
   Public Methods
 * ------------------------ */

/** Return the left singular vectors
@return     U
*/
func (s *SingularValueDecomposition) GetU() *Matrix {
	mat, _ := NewMatrixWithArrays(s.U)
	return mat
}

/** Return the right singular vectors
@return     V
*/
func (s *SingularValueDecomposition) GetV() *Matrix {
	mat, _ := NewMatrixWithMatrix(s.V, s.N, s.N)
	return mat
}

/** Return the one-dimensional array of singular values
@return     diagonal of S.
*/

func (s *SingularValueDecomposition) GetSingularValues() []float64 {
	return s.S
}

/** Return the diagonal matrix of singular values
@return     S
*/
func (s *SingularValueDecomposition) GetS() *Matrix {
	X := NewMatrix(s.N, s.N)

	for i := 0; i < s.N; i++ {
		for j := 0; j < s.N; j++ {
			X.A[i][j] = 0.0
		}
		X.A[i][i] = s.S[i]
	}
	return X
}

/** Two norm
@return     max(S)
*/
func (s *SingularValueDecomposition) Norm2() float64 {
	return s.S[0]
}

/** Two norm condition number
@return     max(S)/min(S)
*/
func (s *SingularValueDecomposition) Cond() float64 {
	return s.S[0] / s.S[minInt(s.M, s.N)-1]
}

/** Effective numerical matrix rank
@return     Number of nonnegligible singular values.
*/
func (s *SingularValueDecomposition) Rank() int {

	eps := math.Pow(2.0, -52.0)
	tol := float64(maxInt(s.M, s.N)) * s.S[0] * eps
	r := 0
	for i := 0; i < len(s.S); i++ {
		if s.S[i] > tol {
			r++
		}
	}
	return r
}
