package model

import (
	"encoding/json"
	"facedetection/facedetector"
	"fmt"
	"github.com/KatyBlumer/Go-Eigenface-Face-Distance/eigenface"
	"github.com/KatyBlumer/Go-Eigenface-Face-Distance/faceimage"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"sync"
)

type User struct {
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
}

type UserFace struct {
	User           User                   `json:"user"`
	Faces          []eigenface.FaceVector `json:"eigenfaces"`
	AverageFace    eigenface.FaceVector   `json:"average_face"`
	TrainingImages []string               `json:"training_images"`
}

type UsersLib struct {
	UsersFace map[string]*UserFace `json:"users_lib"`
}

var (
	DataPath        = "Data"
	loadUserLibOnce sync.Once
	userLibLock     sync.Mutex
	usersLib        *UsersLib
)

func GetUsersLib() *UsersLib {
	loadUserLibOnce.Do(func() {
		usersLib = &UsersLib{}
		usersLib.UsersFace = make(map[string]*UserFace, 0)
		_, err := os.Stat(DataPath)
		if err != nil {
			os.MkdirAll(DataPath, os.ModePerm)
		}
	})
	usersLib.save()
	return usersLib
}
func (ul *UsersLib) AddUserFace(u *UserFace) {
	ul.UsersFace[u.GetKey()] = u
	ul.save()
}

func (u *UsersLib) save() {
	userLibLock.Lock()
	defer userLibLock.Unlock()
	f, err := os.Create(DataPath + string(filepath.Separator) + "data_library.json")
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	defer f.Close()
	err = json.NewEncoder(f).Encode(u)
	if err != nil {
		fmt.Println(err.Error())
		return
	}
}

func NewUserFace() *UserFace {
	return &UserFace{}
}

func (u *UserFace) GetKey() string {
	return u.User.FirstName + "." + u.User.LastName
}

func (u *UserFace) DetectFaces(images []string) {
	userBasePath := DataPath + string(filepath.Separator) + u.GetKey()
	_, err := os.Stat(userBasePath)
	if err != nil {
		os.MkdirAll(userBasePath, os.ModePerm)
	}
	for _, img := range images {
		fd := facedetector.NewFaceDetector(img)
		filespaths := fd.DrawImageInDirectory(userBasePath)
		u.TrainingImages = append(u.TrainingImages, filespaths...)
	}
	return
}

func (u *UserFace) SaveAverageFace() {
	img := faceimage.ToImage(u.AverageFace)
	out, err := os.Create(DataPath + string(filepath.Separator) + u.GetKey() + string(filepath.Separator) + "average.png")
	if err != nil {
		fmt.Println(err)
		return
	}

	err = png.Encode(out, img)
	if err != nil {
		fmt.Println(err)
		return
	}
}
func (u *UserFace) SaveNormalizedFaces() {
	for index, face := range u.Faces {
		img := faceimage.ToImage(face)
		out, err := os.Create(DataPath + string(filepath.Separator) + u.GetKey() + string(filepath.Separator) + strconv.Itoa(index) + "_normalized.png")
		if err != nil {
			fmt.Println(err)
			return
		}
		defer out.Close()

		err = png.Encode(out, img)
		if err != nil {
			fmt.Println(err)
			return
		}
	}
}

func (u *UserFace) TrainFaces() {
	faces := make([]eigenface.FaceVector, len(u.TrainingImages))
	for index, imagePath := range u.TrainingImages {
		faces[index] = faceimage.ToVector(imagePath)
	}

	u.Faces = eigenface.Normalize(faces)
	u.AverageFace = eigenface.Average(faces)

	return
}
