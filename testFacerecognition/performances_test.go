package testFacerecognition

import (
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
