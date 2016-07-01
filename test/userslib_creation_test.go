package test

import (
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
