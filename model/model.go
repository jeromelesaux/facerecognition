package model

import (
	"encoding/json"
	"facedetection/facedetector"
	"facerecognition/eigen"
	"fmt"
	"image"
	"image/png"
	"math"
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

func SumPixels(face eigenface.FaceVector) float64 {
	sum := 0.
	for i := 0; i < (face.Width * face.Height); i++ {
		sum += face.Pixels[i]
	}
	return math.Abs(sum / float64(face.Width*face.Height) / 0xffff)
}

func (u *UserFace) SaveAverageFace() {
	img := ToImage(u.AverageFace)
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

func SaveImageTo(img *image.Gray16, path string) {
	out, err := os.Create(path)
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

func (u *UserFace) SaveNormalizedFaces() {
	for index, face := range u.Faces {
		img := ToImage(face)
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
		faces[index] = ToVector(imagePath)
	}

	u.Faces = eigenface.Normalize(faces)
	u.AverageFace = eigenface.Average(faces)
	u.SaveAverageFace()
	u.SaveNormalizedFaces()

	return
}

func (u *UsersLib) RecognizeFace(imagePath string) {
	f := facedetector.NewFaceDetector(imagePath)
	_, err := os.Stat("tmp")
	if err != nil {
		os.MkdirAll("tmp", os.ModePerm)
	}
	filespaths := f.DrawImageInDirectory("tmp")
	facesDetected := make([]eigenface.FaceVector, 0)
	for _, filePath := range filespaths {
		fvector := ToVector(filePath)
		facesDetected = append(facesDetected, fvector)
	}
	averageFace := eigenface.Average(facesDetected)

	for key, userFace := range u.UsersFace {

		result := eigenface.Difference(userFace.AverageFace, averageFace)
		fmt.Println(strconv.Itoa(result.Height) + ":" + strconv.Itoa(result.Width))
		if result.Width == 0 && result.Height == 0 {
			fmt.Println(imagePath + " contains " + key)
		}

	}
}
func (u *UsersLib) CompareFace(facepath string) {
	facevector := ToVector(facepath)
	facesVector := make([]eigenface.FaceVector, 0)
	facesVector = append(facesVector, facevector)
	averageFace := eigenface.Average(facesVector)

	SaveImageTo(ToImage(averageFace), "tmp/average_barrack.png")
	//for key, userFace := range u.UsersFace {
	//
	//	fvs := eigenface.NewFaceVectors(userFace.AverageFace.Height, userFace.AverageFace.Width, len(userFace.Faces))
	//	fvs.Pixels = userFace.Faces
	//	fvs.Train()
	//
	//	distance := fvs.ComputeDistance(facevector)
	//
	//	fmt.Println(key + " Distance : " + strconv.FormatFloat(distance, 'b', -1, 64))
	//	//distance := eigenface.ComputeDistance(userFace.AverageFace.Pixels, averageFace.Pixels)
	//	////distance := userFace.LevenshteinDistance(averageFace)
	//	//fmt.Println(key + " Distance : " + strconv.FormatFloat(distance, 'b', -1, 64))
	//	////faceVectorAv := eigenface.Average(faces)
	//	////fmt.Println(key + " " + faceVectorAv.ToString())
	//	//diffVector := eigenface.Difference(userFace.AverageFace, averageFace)
	//	//distance = eigenface.ComputeDistance(diffVector.Pixels, userFace.AverageFace.Pixels)
	//	//fmt.Println(key + " diff Distance : " + strconv.FormatFloat(distance, 'b', -1, 64))
	//}
}

func (u *UserFace) LevenshteinDistance(face1 eigenface.FaceVector) float64 {
	s1 := len(face1.Pixels)
	s2 := len(u.AverageFace.Pixels)
	if s1 == 0 {
		return 0
	}
	if s2 == 0 {
		return 0
	}
	matrix1 := make([]float64, s1)
	matrix2 := make([]float64, s2)

	for i := 0; i < s2; i++ {
		matrix1[i] = float64(i)
	}
	for i := 0; i < s1-1; i++ {
		matrix2[0] = float64(i + 1)
		for j := 0; j < s2-1; j++ {
			cost := 1.
			if face1.Pixels[i] == u.AverageFace.Pixels[j] {
				cost = 0.
			}
			matrix2[j+1] = MIN(matrix2[j]+1, matrix1[j+1]+1, matrix1[j]+cost)
		}

		for j := 0; j < len(matrix2); j++ {
			matrix1[j] = matrix2[j]
		}

	}

	return matrix2[s2-1]
}

func MIN(a, b, c float64) float64 {
	if a > b {
		if b > c {
			return c
		} else {
			return b
		}
	} else {
		if a > c {
			return c
		} else {
			return a
		}
	}

}
