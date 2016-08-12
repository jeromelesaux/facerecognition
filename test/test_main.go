package main

import (
	"path/filepath"
	"fmt"
	"strconv"
	"facerecognition/model"
)

func main() {
	facesvectors := model.NewFacesVectors()
	faces := facesvectors.DetectFaces("/home/jlesaux/workspace/go_dev/src/facerecognition/test/images/trainingset-barrack.png")
	person := facesvectors.AddUser("Barrack", "Obama", faces)
	person.FaceVectors.Train()
	img := model.ToImage(person.FaceVectors.Mean)
	model.SaveImageTo(img, model.DataPath+string(filepath.Separator)+"tmp"+string(filepath.Separator)+person.User.Key()+"-average.png")
	face := facesvectors.DetectFaces("/home/jlesaux/workspace/go_dev/src/facerecognition/test/images/barack.png")
	reconstructed, distance := person.FaceVectors.ComputeDistance(face[0])
	img = model.ToImage(reconstructed)
	model.SaveImageTo(img, model.DataPath+string(filepath.Separator)+"tmp"+string(filepath.Separator)+person.User.Key()+"-reconstructed.png")
	fmt.Println("Barrack training distance is ", strconv.FormatFloat(distance, 'f', 10, 64))
}
