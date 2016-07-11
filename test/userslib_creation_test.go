package test

import (
	"facerecognition/eigen"
	"facerecognition/model"
	_ "image/png"
	"testing"
)

func TestUsersLibCreation(t *testing.T) {
	ul := model.GetUsersLib()
	uf := model.NewUserFace()
	uf.User.LastName = "Doe"
	uf.User.FirstName = "John"

	images := make([]string, 0)
	images = append(images, "images/trainingset.png")

	uf.DetectFaces(images)
	uf.TrainFaces()
	ul.AddUserFace(uf)
}

func TestClooneyDetectiong(t *testing.T) {
	ul := model.GetUsersLib()
	uf := model.NewUserFace()
	uf.User.LastName = "Clooney"
	uf.User.FirstName = "George"

	images := make([]string, 0)
	images = append(images, "images/clooney_set.png")

	uf.DetectFaces(images)
	uf.TrainFaces()
	ul.AddUserFace(uf)
}

func TestBarrackDetectiong(t *testing.T) {
	ul := model.GetUsersLib()
	uf := model.NewUserFace()
	uf.User.LastName = "Obama"
	uf.User.FirstName = "Barrack"

	images := make([]string, 0)
	images = append(images, "images/trainingset-barrack.png")

	uf.DetectFaces(images)
	uf.TrainFaces()
	ul.AddUserFace(uf)
}

func TestCompareBarrack(t *testing.T) {
	ul := model.GetUsersLib()
	barrackVector := model.ToVector("images/barrack_face.png")
	for key, person := range ul.UsersFace {
		average := eigenface.Difference(barrackVector, person.AverageFace)

		i := model.ToImage(average)
		model.SaveImageTo(i, "tmp/barrack_"+key+".png")
		model.SaveImageTo(model.ToImage(person.AverageFace), "tmp/"+key+".png")

	}
}
