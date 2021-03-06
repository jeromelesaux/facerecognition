package testFacerecognition

import (
	"github.com/jeromelesaux/facerecognition/logger"
	"github.com/jeromelesaux/facerecognition/model"
	"strconv"
	"testing"
)

func TestS1Trainer(t *testing.T) {
	m := &model.CosineDissimilarity{}
	trainer := model.NewTrainerArgs("PCA", 1, 3, m.GetDistance)
	//trainer.FeatureType = "PCA"
	//trainer.K = 1
	//trainer.NumOfComponents = 3
	//cosineDissimilarity := &model.CosineDissimilarity{}
	//trainer.Metric = cosineDissimilarity.GetDistance

	filepaths := []string{"faces/s2/1.pgm", "faces/s2/2.pgm", "faces/s2/3.pgm"}
	for _, value := range filepaths {
		logger.Log("Reading file " + value)
		mat := model.ToMatrix(value)

		logger.Log("Read " + value + " with width " + strconv.Itoa(mat.M) + " and height " + strconv.Itoa(mat.N))
		trainer.Add(mat.Vectorize(), "john")
	}

	filepaths2 := []string{"faces/s4/1.pgm", "faces/s4/2.pgm", "faces/s4/3.pgm"}
	for _, value := range filepaths2 {
		logger.Log("Reading file " + value)
		mat := model.ToMatrix(value)
		logger.Log("Read " + value + " with width " + strconv.Itoa(mat.M) + " and height " + strconv.Itoa(mat.N))
		trainer.Add(mat.Vectorize(), "smith")
	}
	trainer.Train()
	mat := model.ToMatrix("faces/s2/4.pgm")
	personFound1, distance := trainer.Recognize(mat.Vectorize())
	logger.Log("found :" + personFound1 + " distance : " + strconv.FormatFloat(distance, 'f', 2, 32))
	if "john" != personFound1 {
		t.Fatal("Expected john and " + personFound1 + " found")
	}
	if distance < .9 {
		t.Fatal("Expected more than .9 obtained : " + strconv.FormatFloat(distance, 'f', 2, 32))
	}
	mat2 := model.ToMatrix("faces/s4/4.pgm")
	personFound2, distance := trainer.Recognize(mat2.Vectorize())
	logger.Log("found :" + personFound2 + " distance : " + strconv.FormatFloat(distance, 'f', 2, 32))
	if "smith" != personFound2 {
		t.Fatal("Expected john and " + personFound2 + " found")
	}
	if distance < .9 {
		t.Fatal("Expected more than .9 obtained : " + strconv.FormatFloat(distance, 'f', 2, 32))
	}
}
