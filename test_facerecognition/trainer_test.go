package test_facerecognition

import (
	"facerecognition/logger"
	"facerecognition/model"
	"strconv"
	"testing"
)

func TestS1Trainer(t *testing.T) {
	filepaths := []string{"faces/s2/1.pgm", "faces/s2/2.pgm", "faces/s2/3.pgm", "faces/s2/4.pgm"}
	for _, value := range filepaths {
		logger.Log("Reading file " + value)
		width, height, v := model.ToVector(value)
		logger.Log("Read " + value + " with width " + strconv.Itoa(width) + " and height " + strconv.Itoa(height))
		model.ToMatrix(width, height, v)
	}
}
