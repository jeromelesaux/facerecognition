package testFacerecognition

import (
	"github.com/jeromelesaux/facerecognition/logger"
	"github.com/jeromelesaux/facerecognition/model"
	"image"
	"os"
	"strconv"
	"testing"
)

func TestBarrackTrainer(t *testing.T) {
	//m := &model.CosineDissimilarity{}
	//trainer := model.NewTrainerArgs("PCA", 1, 3, m.GetDistance)
	userslib := model.GetUsersLib()
	f, _ := os.Open("images/trainingset-barrack.png")
	img, _, _ := image.Decode(f)
	mats, files := userslib.FindFace(&img)
	if len(mats) == 0 {
		t.Fatal("expected len mats > to 0")
	}
	if len(files) == 0 {
		t.Fatal("expected len files > to 0")
	}
	m := &model.CosineDissimilarity{}
	trainer := model.NewTrainerArgs("PCA", 1, 3, m.GetDistance)
	for _, value := range files {
		logger.Log("Reading file " + value)
		mat := model.ToMatrix(value)

		logger.Log("Read " + value + " with width " + strconv.Itoa(mat.M) + " and height " + strconv.Itoa(mat.N))
		trainer.Add(mat.Vectorize(), "barrack")
	}

}
