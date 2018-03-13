package testFacerecognition

import (
	"github.com/jeromelesaux/facerecognition/algorithm"
	"github.com/jeromelesaux/facerecognition/model"
	"testing"
)

func init() {
	model.SetAndLoad("../config.json")
}

func BenchmarkPerformanceRecognition(b *testing.B) {
	l := model.GetFaceRecognitionLib()
	tr := l.GetTrainer(model.PCAFeatureType)
	tr.Train()
}

func TestTimesMatrices(t *testing.T) {
	m1 := [][]float64{{1., 2., 3.}, {1., 2., 3.}, {1., 2., 3.}}
	m1m, err := algorithm.NewMatrixWithArrays(m1)
	if err != nil {
		t.Fatalf("this error was not  expected %v", err)
	}
	m2m, err := algorithm.NewMatrixWithArrays(m1)
	if err != nil {
		t.Fatalf("this error was not  expected %v", err)
	}
	m3m := m1m.TimesMatrix(m2m)
	for i := 0; i < m3m.N; i++ {
		for j := 0; j < m3m.M; j++ {
			switch j {
			case 0:
				if m3m.A[i][j] != 6.0 {
					t.Fatalf("expected value 6.0 and gets m3m[%d][%d]:%f", i, j, m3m.A[i][j])
				}
			case 1:
				if m3m.A[i][j] != 12.0 {
					t.Fatalf("expected value 12.0 and gets m3m[%d][%d]:%f", i, j, m3m.A[i][j])
				}
			case 2:
				if m3m.A[i][j] != 18.0 {
					t.Fatalf("expected value 18.0 and gets m3m[%d][%d]:%f", i, j, m3m.A[i][j])
				}
			}
		}
	}
	t.Log(m3m.Tostring())

}
