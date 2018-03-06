package model

import (
	"github.com/cnf/structhash"
	"github.com/jeromelesaux/facerecognition/algorithm"
	"github.com/jeromelesaux/facerecognition/logger"
)

type LPP struct {
	FeatureExtraction *FeatureExtraction
}

func NewLPP(trainingSet []*algorithm.Matrix, labels []string, numOfComponents int) *LPP {
	lpp := &LPP{FeatureExtraction: NewFeatureExtraction()}
	//n := len(trainingSet)
	tempSet := make(map[string]int)
	for i := 0; i < len(labels); i++ {
		if _, ok := tempSet[labels[i]]; !ok {
			tempSet[labels[i]] = 1
		} else {
			tempSet[labels[i]] += 1
		}
	}
	c := len(tempSet)
	// process in PCA
	pca := NewPCA(trainingSet, labels, numOfComponents)
	//construct the nearest neighbor graph
	s := construcNearestNeighborGraph(pca.FeatureExtraction.ProjectedTrainingSet)
	dd := constructD(s)
	l := dd.Minus(s)
	//reconstruct the trainingSet into required X;
	x := constructTrainingMatrix(pca.FeatureExtraction.ProjectedTrainingSet)
	xlxt := x.TimesMatrix(l).TimesMatrix(x.Transpose())
	xdxt := x.TimesMatrix(dd).TimesMatrix(x.Transpose())

	//calculate the eignevalues and eigenvectors of (XDXT)^-1 * (XLXT)
	targetForEigen := xdxt.Inverse().TimesMatrix(xlxt)
	feature := targetForEigen.Eig()
	d := feature.Getd()
	if len(d) < c-1 {
		logger.Logf("Ensure that the number of eigenvalues is larger than c - 1 (%d,%d)", len(d), (c - 1))
		return lpp
	}
	indexes := GetIndexesOfKEigenvalues(d, len(d))
	eigenVectors := feature.GetV()
	selectedEigenVectors := eigenVectors.GetMatrix2(0, eigenVectors.RowsDimension()-1, indexes)
	lpp.FeatureExtraction.W = pca.FeatureExtraction.W.TimesMatrix(selectedEigenVectors)
	lpp.FeatureExtraction.ProjectedTrainingSet = make([]*ProjectedTrainingMatrix, 0)
	//Construct projectedTrainingMatrix
	for i := 0; i < len(trainingSet); i++ {
		ptm := NewProjectedTrainingMatrix(lpp.FeatureExtraction.W.Transpose().TimesMatrix(trainingSet[i].Minus(pca.FeatureExtraction.MeanMatrix)), labels[i])
		lpp.FeatureExtraction.ProjectedTrainingSet = append(lpp.FeatureExtraction.ProjectedTrainingSet, ptm)
	}

	lpp.FeatureExtraction.MeanMatrix = pca.FeatureExtraction.MeanMatrix
	return lpp
}

func construcNearestNeighborGraph(input []*ProjectedTrainingMatrix) *algorithm.Matrix {
	size := len(input)
	s := algorithm.NewMatrix(size, size)
	euclidean := &Euclidean{}
	trainArray := NewSliceProjectedTrainingMatrix(input)
	for i := 0; i < size; i++ {
		neighbors := FindKNN(trainArray, input[i].Matrix, 3, euclidean.GetDistance)
		hashIndexes := hashMatrices(neighbors)
		for j := 0; j < len(neighbors); j++ {
			hash := string(structhash.Md5(input[i], 1))
			if index, ok := hashIndexes[hash]; ok {
				s.A[i][index] = 1.0
				s.A[index][i] = 1.0
			}
		}
	}
	return s
}

func hashMatrices(input []*ProjectedTrainingMatrix) map[string]int {
	hashIndexes := make(map[string]int, len(input))
	for i := 0; i < len(input); i++ {
		hashIndexes[string(structhash.Md5(input[i], 1))] = i
	}
	return hashIndexes
}

func constructD(s *algorithm.Matrix) *algorithm.Matrix {
	size := s.RowsDimension()
	d := algorithm.NewMatrix(size, size)
	for i := 0; i < size; i++ {
		temp := 0.0
		for j := 0; j < size; j++ {
			temp += s.A[j][i]
		}
		d.A[i][i] = temp
	}
	return d
}

func constructTrainingMatrix(input []*ProjectedTrainingMatrix) *algorithm.Matrix {
	row := input[0].Matrix.RowsDimension()
	column := len(input)
	x := algorithm.NewMatrix(row, column)

	for i := 0; i < column; i++ {
		x.SetMatrix(0, row-1, i, i, input[i].Matrix)
	}

	return x
}
