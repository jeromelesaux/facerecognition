package model

import (
	"github.com/jeromelesaux/facerecognition/algorithm"
)

type ProjectedTrainingMatrix struct {
	Matrix   *algorithm.Matrix
	Label    string
	Distance float64
}

func NewSliceProjectedTrainingMatrix(input []*ProjectedTrainingMatrix) []*ProjectedTrainingMatrix {
	ptm := make([]*ProjectedTrainingMatrix, 0)
	for _, v := range input {
		ptm = append(ptm, NewProjectedTrainingMatrix(v.Matrix, v.Label))
	}
	return ptm
}

func NewProjectedTrainingMatrix(m *algorithm.Matrix, l string) *ProjectedTrainingMatrix {
	return &ProjectedTrainingMatrix{Matrix: m, Label: l}
}
