package model

import (
	"facerecognition/algorithm"
)

type Trainer struct {
	Metric            func(a, b *algorithm.Matrix) float64
	FeatureType       string
	FeatureExtraction *FeatureExtraction
	NumOfComponents   int
	K                 int
	TrainingSet       []*algorithm.Matrix
	TrainingLabels    []string
	Model             []*ProjectedTrainingMatrix
}

func NewTrainer() *Trainer {
	t := &Trainer{}
	t.TrainingSet = make([]*algorithm.Matrix, 0)
	t.TrainingLabels = make([]string, 0)
	t.Model = make([]*ProjectedTrainingMatrix, 0)
	t.FeatureExtraction = NewFeatureExtraction()
	return t
}

func NewTrainerArgs(featureType string, k int, numOfComponents int, distanceFunction func(a, b *algorithm.Matrix) float64) *Trainer {
	t := NewTrainer()
	t.FeatureType = featureType
	t.K = k
	t.NumOfComponents = numOfComponents
	t.Metric = distanceFunction
	return t
}

func (t *Trainer) Add(m *algorithm.Matrix, label string) {
	t.TrainingSet = append(t.TrainingSet, m)
	t.TrainingLabels = append(t.TrainingLabels, label)
}

func (t *Trainer) Train() {
	switch t.FeatureType {
	case "PCA":
		p := NewPCA(t.TrainingSet, t.TrainingLabels, t.NumOfComponents)
		t.FeatureExtraction = p.FeatureExtraction
		break
	case "LDA":
		l := NewLDA(t.TrainingSet, t.TrainingLabels, t.NumOfComponents)
		t.FeatureExtraction = l.FeatureExtraction
		break
	case "LPP":
		l := NewLPP(t.TrainingSet, t.TrainingLabels, t.NumOfComponents)
		t.FeatureExtraction = l.FeatureExtraction
	}

	t.Model = t.FeatureExtraction.ProjectedTrainingSet
}
func (t *Trainer) Recognize(matrix *algorithm.Matrix) string {
	testCase := t.FeatureExtraction.W.Transpose().TimesMatrix(matrix.Minus(t.FeatureExtraction.MeanMatrix))
	result := AssignLabel(t.Model, testCase, t.K, t.Metric)
	return result
}
