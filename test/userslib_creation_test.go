package test

import (
	"facerecognition/model"
	"fmt"
	_ "image/png"
	"path/filepath"
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
		fmt.Println(key + " : " + strconv.FormatFloat(sum, 'f', 10, 64))
		i := model.ToImage(average)
		model.SaveImageTo(i, "tmp/barrack_"+key+".png")
		model.SaveImageTo(model.ToImage(person.AverageFace), "tmp/"+key+".png")

	}
	fmt.Println(faceFound + " seems to be the person you're looking for.")
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
	facesvectors := model.NewFacesVectors()
	faces := facesvectors.DetectFaces("images/trainingset-barrack.png")
	person := facesvectors.AddUser("Barrack", "Obama", faces)
	person.FaceVectors.Train()
	img := model.ToImage(person.FaceVectors.Mean)
	model.SaveImageTo(img, model.DataPath+string(filepath.Separator)+"tmp"+string(filepath.Separator)+person.User.Key()+"-average.png")
	face := facesvectors.DetectFaces("images/barack.png")
	reconstructed, distance := person.FaceVectors.ComputeDistance(face[0])
	img = model.ToImage(reconstructed)
	model.SaveImageTo(img, model.DataPath+string(filepath.Separator)+"tmp"+string(filepath.Separator)+person.User.Key()+"-reconstructed.png")
	fmt.Println("Barrack training distance is ", strconv.FormatFloat(distance, 'f', 10, 64))
}
