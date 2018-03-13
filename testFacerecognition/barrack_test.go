package testFacerecognition

import (
	"fmt"
	"github.com/jeromelesaux/facerecognition/logger"
	"github.com/jeromelesaux/facerecognition/model"
	"image"
	"os"
	"strconv"
	"testing"
)

func init() {
	model.SetAndLoad("../config.json")
	fl := model.NewFaceRecognitionLib()
	for i := 1; i < 41; i++ {
		key := fmt.Sprintf("s%d.train", i)
		firstname := fmt.Sprintf("s%d", i)
		fl.Items[key] = &model.FaceRecognitionItem{User: model.User{FirstName: firstname, LastName: "train"}}
	}
	fl.Save()
}

func TestBarrackObamaDetection(t *testing.T) {
	userslib := model.GetFaceRecognitionLib()
	f, err := os.Open("images/barack.png")
	if err != nil {
		t.Fatalf("expected image and gets error %v", err)
	}

	img, _, _ := image.Decode(f)
	mats, files := userslib.FindFace(&img)
	if len(mats) == 0 {
		t.Fatal("expected len mats > to 0")
	}
	if len(files) == 0 {
		t.Fatal("expected len files > to 0")
	}
	t.Logf("Return [%d] images.", len(mats))
}

func TestBarrackTrainer(t *testing.T) {
	//m := &model.CosineDissimilarity{}
	//trainer := model.NewTrainerArgs("PCA", 1, 3, m.GetDistance)
	userslib := model.GetFaceRecognitionLib()
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

/*func TestDrawFoundFaces(t *testing.T) {
	lib := model.GetFaceRecognitionLib()
	f, _ := os.Open("images/barack.png")
	img, _, _ := image.Decode(f)
	matrices, _ := lib.FindFace(&img)
	trainer := lib.GetTrainer(model.PCAFeatureType)
	trainer.Train()
	for _, v := range matrices {
		result, distance := trainer.Recognize(v)
		t.Logf("results : %s for distance %f\n", result, distance)
	}

}
*/
