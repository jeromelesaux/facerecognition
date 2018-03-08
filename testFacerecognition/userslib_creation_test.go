package testFacerecognition

import (
	"github.com/jeromelesaux/facedetection/facedetector"
	"github.com/jeromelesaux/facerecognition/model"
	_ "image/png"
	"strconv"
	"testing"
)

//
//func TestUsersLibCreation(t *testing.T) {
//	ul := model.GetUsersLib()
//	uf := model.NewUserFace()
//	uf.User.LastName = "Doe"
//	uf.User.FirstName = "John"
//
//	images := make([]string, 0)
//	images = append(images, "images/trainingset.png")
//
//	uf.DetectFaces(images)
//	uf.TrainFaces()
//	ul.AddUserFace(uf)
//}
//
//func TestClooneyDetectiong(t *testing.T) {
//	ul := model.GetUsersLib()
//	uf := model.NewUserFace()
//	uf.User.LastName = "Clooney"
//	uf.User.FirstName = "George"
//
//	images := make([]string, 0)
//	images = append(images, "images/clooney_set.png")
//
//	uf.DetectFaces(images)
//	uf.TrainFaces()
//	ul.AddUserFace(uf)
//}
//
//func TestBarrackDetectiong(t *testing.T) {
//	ul := model.GetUsersLib()
//	uf := model.NewUserFace()
//	uf.User.LastName = "Obama"
//	uf.User.FirstName = "Barrack"
//
//	images := make([]string, 0)
//	images = append(images, "images/trainingset-barrack.png")
//
//	uf.DetectFaces(images)
//	uf.TrainFaces()
//	ul.AddUserFace(uf)
//}

/*
func TestCompareBarrack(t *testing.T) {
	ul := model.GetUsersLib()
	barrackVector := model.ToVector("images/barrack_face.png")
	sumConserved := 1.
	faceFound := ""
	for key, person := range ul.UsersFace {
		average := eigenface.Difference(barrackVector, person.AverageFace)
		sum := model.SumPixels(average)
		if sum < sumConserved {
			sumConserved = sum
			faceFound = key
		}
		logger.Log(key + " : " + strconv.FormatFloat(sum, 'f', 10, 64))
		i := model.ToImage(average)
		model.SaveImageTo(i, "tmp/barrack_"+key+".png")
		model.SaveImageTo(model.ToImage(person.AverageFace), "tmp/"+key+".png")

	}
	logger.Log(faceFound + " seems to be the person you're looking for.")
}

func TestDetectGeorge(t *testing.T) {
	images := make([]string, 0)
	images = append(images, "images/george-clooney.png")
	uf := model.NewUserFace()
	uf.User.LastName = "Test1"
	uf.User.FirstName = "TEST1"
	uf.DetectFaces(images)
}                     */

//func TestCompareGeorge(t *testing.T) {
//	ul := model.GetUsersLib()
//	ul.RecognizeFace("images/barack.png")
//}

func TestDetectAndTrainBarrack(t *testing.T) {
	ul := model.GetFaceRecognitionLib()
	fc := facedetector.NewFaceDetector("images/trainingset-barrack.png", "haarcascade_frontalface_default.xml")
	barrack := model.NewFaceRecognitionItem()
	barrack.User.FirstName = "Barrack"
	barrack.User.LastName = "Obama"
	user := ul.ImportIntoDB(fc, barrack)
	if len(user.TrainingImages) != 3 {
		t.Fatal("training length does not corresponded expetecd 3 obtenainded " + strconv.Itoa(len(user.TrainingImages)))
	}
	//os.RemoveAll("Data")
}

func TestDetectFaceWomen(t *testing.T) {
	uf := model.NewFaceRecognitionItem()
	uf.DetectFaces([]string{"images/32m54jg7.jpg"})

	t.Logf("%v", uf.User)
}
