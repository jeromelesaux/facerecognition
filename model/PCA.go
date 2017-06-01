package model

import (
	"github.com/jeromelesaux/facerecognition/algorithm"
	"github.com/jeromelesaux/facerecognition/logger"
	"math"
)

type PCA struct {
	FeatureExtraction *FeatureExtraction
}

func NewPCA(trainingSet []*algorithm.Matrix, labels []string, numOfComponents int) *PCA {
	p := &PCA{FeatureExtraction: NewFeatureExtraction()}
	p.FeatureExtraction.TrainingSet = trainingSet
	p.FeatureExtraction.Labels = labels
	p.FeatureExtraction.NumOfComponents = numOfComponents
	p.FeatureExtraction.MeanMatrix = p.GetMean(p.FeatureExtraction.TrainingSet)
	p.FeatureExtraction.W = p.GetFeature(p.FeatureExtraction.TrainingSet, p.FeatureExtraction.NumOfComponents)
	// Construct projectedTrainingMatrix
	p.FeatureExtraction.ProjectedTrainingSet = make([]*ProjectedTrainingMatrix, 0)
	for i := 0; i < len(trainingSet); i++ {
		ptm := NewProjectedTrainingMatrix(p.FeatureExtraction.W.Transpose().TimesMatrix(trainingSet[i].Minus(p.FeatureExtraction.MeanMatrix)), labels[i])
		p.FeatureExtraction.ProjectedTrainingSet = append(p.FeatureExtraction.ProjectedTrainingSet, ptm)
	}
	return p
}

func (p *PCA) GetMean(input []*algorithm.Matrix) *algorithm.Matrix {
	rows := input[0].RowsDimension()
	length := len(input)
	all := algorithm.NewMatrix(rows, 1)
	for i := 0; i < length; i++ {
		all.PlusEqual(input[i])
	}
	return all.Times(1.0 / float64(length))
}

func (p *PCA) GetFeature(input []*algorithm.Matrix, k int) *algorithm.Matrix {
	row := input[0].RowsDimension()
	column := len(input)
	//logger.Log("row,column:"+strconv.Itoa(row)+","+strconv.Itoa(column))
	x := algorithm.NewMatrix(row, column)
	// get eigenvalues and eigenvectors
	for i := 0; i < column; i++ {
		x.SetMatrix(0, row-1, i, i, input[i].Minus(p.FeatureExtraction.MeanMatrix))
	}

	xt := x.Transpose()
	xtx := xt.TimesMatrix(x)
	feature := xtx.Eig()
	d := feature.Getd()

	if len(d) < k {
		logger.Log("number of eigenvalues is less than K")
		return &algorithm.Matrix{}
	}
	indexes := GetIndexesOfKEigenvalues(d, k)
	eigenVectors := x.TimesMatrix(feature.GetV())
	selectedEigenVectors := eigenVectors.GetMatrix2(0, eigenVectors.RowsDimension()-1, indexes)
	// normalize the eigenvectors
	row = selectedEigenVectors.RowsDimension()
	column = selectedEigenVectors.ColumnsDimension()
	for i := 0; i < column; i++ {
		temp := 0.0
		for j := 0; j < row; j++ {
			temp += math.Pow(selectedEigenVectors.A[j][i], 2)
		}
		temp = math.Sqrt(temp)
		for j := 0; j < row; j++ {
			selectedEigenVectors.A[j][i] /= temp
		}
	}
	return selectedEigenVectors
}
