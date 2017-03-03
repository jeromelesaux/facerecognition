package testFacerecognition

import (
	"facerecognition/logger"
	"facerecognition/model"
	"strconv"
	"testing"
)

func TestS1Trainer(t *testing.T) {
	trainer := model.NewTrainerArgs("PCA", 1, 3, &model.CosineDissimilarity{}.GetDistance)
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
	personFound1 := trainer.Recognize(mat.Vectorize())
	logger.Log("found :" + personFound1)

	mat2 := model.ToMatrix("faces/s4/4.pgm")
	personFound2 := trainer.Recognize(mat2.Vectorize())
	logger.Log("found :" + personFound2)

}
