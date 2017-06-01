package model

import (
	"github.com/jeromelesaux/facerecognition/algorithm"
)

type ProjectedTrainingMatrix struct {
	Matrix   *algorithm.Matrix
	Label    string
	Distance float64
}

func NewProjectedTrainingMatrix(m *algorithm.Matrix, l string) *ProjectedTrainingMatrix {
	return &ProjectedTrainingMatrix{Matrix: m, Label: l}
}
