package algorithm

import (
	"github.com/jeromelesaux/facerecognition/logger"
	"math"
)

type QRDecomposition struct {
	QR    [][]float64 // Array for internal storage of decomposition.
	M     int         //column dimension.
	N     int         //row dimension.
	Rdiag []float64   //Array for internal storage of diagonal of R.
}

func NewQRDecomposition(a *Matrix) *QRDecomposition {
	qr := &QRDecomposition{M: a.RowsDimension(), N: a.ColumnsDimension()}
	copy(qr.QR, a.A)
	qr.Rdiag = make([]float64, qr.N)
	// Main loop.
	for k := 0; k < qr.N; k++ {
		// Compute 2-norm of k-th column without under/overflow.
		nrm := 0.0
		for i := k; i < qr.M; i++ {
			nrm = math.Hypot(nrm, qr.QR[i][k])
		}

		if nrm != 0.0 {
			// Form k-th Householder vector.
			if qr.QR[k][k] < 0 {
				nrm = -nrm
			}
			for i := k; i < qr.M; i++ {
				qr.QR[i][k] /= nrm
			}
			qr.QR[k][k] += 1.0

			// Apply transformation to remaining columns.
			for j := k + 1; j < qr.N; j++ {
				s := 0.0
				for i := k; i < qr.M; i++ {
					s += qr.QR[i][k] * qr.QR[i][j]
				}
				s = -s / qr.QR[k][k]
				for i := k; i < qr.M; i++ {
					qr.QR[i][j] += s * qr.QR[i][k]
				}
			}
		}
		qr.Rdiag[k] = -nrm
	}

	return qr
}

func (q *QRDecomposition) IsFullRank() bool {
	for j := 0; j < q.N; j++ {
		if q.Rdiag[j] == 0 {
			return false
		}
	}
	return true
}

func (q *QRDecomposition) Solve(b *Matrix) *Matrix {
	if b.RowsDimension() != q.M {
		logger.Log("Matrix row dimensions must agree.")
		return &Matrix{}
	}
	if !q.IsFullRank() {
		logger.Log("Matrix is rank deficient.")
		return &Matrix{}
	}
	// Copy right hand side
	nx := b.ColumnsDimension()
	// Compute Y = transpose(Q)*B
	for k := 0; k < q.N; k++ {
		for j := 0; j < nx; j++ {
			s := 0.0
			for i := k; i < q.M; i++ {
				s += q.QR[i][k] * b.A[i][j]
			}
			s = -s / q.QR[k][k]
			for i := k; i < q.M; i++ {
				b.A[i][j] += s * q.QR[i][k]
			}
		}
	}
	// Solve R*X = Y;
	for k := q.N - 1; k >= 0; k-- {
		for j := 0; j < nx; j++ {
			b.A[k][j] /= q.Rdiag[k]
		}
		for i := 0; i < k; i++ {
			for j := 0; j < nx; j++ {
				b.A[i][j] -= b.A[k][j] * q.QR[i][k]
			}
		}
	}
	mat, _ := NewMatrixWithMatrix(b.A, q.N, nx)
	return mat.GetMatrix3(0, q.N-1, 0, nx-1)
	//return (new Matrix(X, n, nx).getMatrix(0, n-1, 0, nx-1));
}
