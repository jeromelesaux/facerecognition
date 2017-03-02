package model

import (
	"facerecognition/algorithm"
	"facerecognition/logger"
)

type LDA struct {
	FeatureExtraction *FeatureExtraction
}

func NewLDA(trainingSet []*algorithm.Matrix, labels []string, numOfComponents int) *LDA {
	l := &LDA{FeatureExtraction: NewFeatureExtraction()}
	n := len(trainingSet)
	tempSet := make(map[string]int)
	for i := 0; i < len(labels); i++ {
		if _, ok := tempSet[labels[i]]; !ok {
			tempSet[labels[i]] = 1
		} else {
			tempSet[labels[i]] += 1
		}
	}
	c := len(tempSet)
	if numOfComponents < n-c {
		logger.Log("the input components is smaller than n - c!")
		return l
	}
	if n < 2*c {
		logger.Log("n is smaller than 2c!")
		return l
	}
	// process in PCA
	pca := NewPCA(trainingSet, labels, n-c)
	// classify
	meanTotal := algorithm.NewMatrix(n-c, 1)
	mmap := make(map[string][]algorithm.Matrix, 0)
	pcaTrain := pca.FeatureExtraction.ProjectedTrainingSet
	for i := 0; i < len(pcaTrain); i++ {
		key := pcaTrain[i].Label
		meanTotal.PlusEqual(pcaTrain[i].Matrix)
		if _, ok := mmap[key]; ok {
			temp := make([]algorithm.Matrix, 0)
			temp = append(temp, *pcaTrain[i].Matrix)
			mmap[key] = temp
		} else {
			temp := mmap[key]
			temp = append(temp, *pcaTrain[i].Matrix)
			mmap[key] = temp
		}
	}
	meanTotal.Times(1.0 / float64(n))

	sw := algorithm.NewMatrix(n-c, n-c)
	sb := algorithm.NewMatrix(n-c, n-c)
	for key, _ := range mmap {
		matrixWithinThatClass := mmap[key]
		meanOfCurrentClass := l.GetMean(matrixWithinThatClass)
		for i := 0; i < len(matrixWithinThatClass); i++ {
			temp1 := matrixWithinThatClass[i].Minus(meanOfCurrentClass)
			temp1 = temp1.TimesMatrix(temp1.Transpose())
			sw.PlusEqual(temp1)
		}
		temp := meanOfCurrentClass.Minus(meanTotal)
		temp = temp.TimesMatrix(temp.Transpose()).Times(float64(len(matrixWithinThatClass)))
		sb.PlusEqual(temp)

	}
	// calculate the eigenvalues and vectors of Sw^-1 * Sb
	targetForEigen := sw.Inverse().TimesMatrix(sb)
	feature := targetForEigen.Eig()
	d := feature.Getd()
	if len(d) < c-1 {
		logger.Log("Ensure that the number of eigenvalues is larger than c - 1")
		return l
	}
	indexes := GetIndexesOfKEigenvalues(d, c-1)
	eigenVectors := feature.GetV()
	selectedEigenVectors := eigenVectors.GetMatrix2(0, eigenVectors.RowsDimension(), indexes)
	l.FeatureExtraction.W = pca.FeatureExtraction.W.TimesMatrix(selectedEigenVectors)
	// Construct projectedTrainingMatrix
	l.FeatureExtraction.ProjectedTrainingSet = make([]*ProjectedTrainingMatrix, 0)
	for i := 0; i < len(trainingSet); i++ {
		ptm := NewProjectedTrainingMatrix(l.FeatureExtraction.W.Transpose().TimesMatrix(trainingSet[i].Minus(pca.FeatureExtraction.MeanMatrix)), labels[i])
		l.FeatureExtraction.ProjectedTrainingSet =append(l.FeatureExtraction.ProjectedTrainingSet, ptm)
	}
	l.FeatureExtraction.MeanMatrix = pca.FeatureExtraction.MeanMatrix
	return l
}

func (l *LDA) GetMean(m []algorithm.Matrix) *algorithm.Matrix {
	num := len(m)
	row := m[0].RowsDimension()
	column := m[0].ColumnsDimension()
	if column != 1 {
		logger.Log("expected column does not equal to 1!")
		return nil
	}
	mean := algorithm.NewMatrix(row, column)
	for i := 0; i < num; i++ {
		mean.PlusEqual(&m[i])
	}
	mean.Times(1.0 / float64(num))
	return mean
}
