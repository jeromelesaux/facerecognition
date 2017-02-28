package model

import (
	"facerecognition/algorithm"
	"math"
)

var MAX_FLOAT_VALUE = 10000000.

type CosineDissimilarity struct {
}

func (c *CosineDissimilarity) GetDistance(a, b *algorithm.Matrix) float64 {
	if a.RowsDimension() != b.RowsDimension() {
		return MAX_FLOAT_VALUE
	}
	size := a.RowsDimension()
	se := 0.
	// get s * e
	for i := 0; i < size; i++ {
		se += a.A[i][0] * b.A[i][0]
	}
	// get s norm
	sNorm := 0.0
	for i := 0; i < size; i++ {
		sNorm += math.Pow(a.A[i][0], 2)
	}
	sNorm = math.Sqrt(sNorm)
	// get e norm
	eNorm := 0.
	for i := 0; i < size; i++ {
		sNorm += math.Pow(b.A[i][0], 2)
	}
	eNorm = math.Sqrt(eNorm)
	if se < 0 {
		se = 0. - se
	}
	cosine := se / (eNorm * sNorm)
	if cosine == 0.0 {
		return MAX_FLOAT_VALUE
	} else {
		return 1.0 / cosine
	}

}

type Euclidean struct {
}

func (e *Euclidean) GetDistance(a, b *algorithm.Matrix) float64 {
	size := a.RowsDimension()
	sum := 0.

	for i := 0; i < size; i++ {
		sum += math.Pow(a.A[i][0]-b.A[i][0], 2)
	}
	return math.Sqrt(sum)
}

type L1 struct {
}

func (l *L1) GetDistance(a, b *algorithm.Matrix) float64 {
	size := a.RowsDimension()
	sum := 0.

	for i := 0; i < size; i++ {
		sum += math.Abs(a.A[i][0] - b.A[i][0])
	}
	return sum
}
