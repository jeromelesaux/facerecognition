package model

import (
	"facerecognition/algorithm"
	"sort"
	"strconv"
)

type FeatureExtraction struct {
	TrainingSet          []*algorithm.Matrix
	Labels               []string
	NumOfComponents      int
	MeanMatrix           *algorithm.Matrix
	W                    *algorithm.Matrix
	ProjectedTrainingSet []*ProjectedTrainingMatrix
}

func NewFeatureExtraction() *FeatureExtraction {
	return &FeatureExtraction{
		NumOfComponents: 0,
		//TrainingSet:make([]*algorithm.Matrix,0),
		Labels: make([]string, 0),
		//ProjectedTrainingSet:make([]*ProjectedTrainingMatrix,0),
	}

}

type mix struct {
	Index int
	Value float64
}

func NewMix(i int, v float64) *mix {
	return &mix{Index: i, Value: v}
}

type MixArray struct {
	Mixes []*mix
}

func NewMixArrays(n int) *MixArray {
	m := &MixArray{}
	m.Mixes = make([]*mix, n)
	return m
}

func (m *MixArray) Len() int {
	return len(m.Mixes)
}

func (m *MixArray) Tostring() string {
	str := ""
	for _, value := range m.Mixes {
		str += strconv.FormatFloat(value.Value, 'f', 16, 32) + " "
	}
	return str
}

func (m *MixArray) Less(i, j int) bool {
	return m.Mixes[i].Value > m.Mixes[j].Value
}

func (m *MixArray) Swap(i, j int) {
	m.Mixes[i], m.Mixes[j] = m.Mixes[j], m.Mixes[i]
}

func GetIndexesOfKEigenvalues(d []float64, k int) []int {
	mixes := NewMixArrays(len(d))
	for i := 0; i < len(d); i++ {
		mixes.Mixes[i] = NewMix(i, d[i])
	}

	sort.Sort(mixes)

	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = mixes.Mixes[i].Index
	}
	return result
}
