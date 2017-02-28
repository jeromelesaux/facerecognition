package model

import (
	"encoding/json"
	"facedetection/facedetector"
	"facerecognition/eigen"
	"facerecognition/logger"
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

func (u *User) ToString() string {
	return "firstname :" + u.FirstName + " and lastname :" + u.LastName
}

func (u *User) Key() string {
	return u.FirstName + "." + u.LastName
}

type UserFace struct {
	User           User        `json:"user"`
	Faces          [][]float64 `json:"eigenfaces"`
	AverageFace    []float64   `json:"average_face"`
	TrainingImages []string    `json:"training_images"`
	FacesDetected  [][]float64 `json:"faces_detected"`
}

type UsersLib struct {
	UsersFace map[string]*UserFace `json:"users_lib"`
}

type PersonVectors struct {
	User        *User                  `json:"user"`
	FaceVectors *eigenface.FaceVectors `json:"faces_vectors"`
}

type FacesVectors struct {
	FacesVectors map[string]*PersonVectors `json:"personns_faces_vectors"`
	Count        int                       `json:"faces_count"`
}

var (
	DataPath        = "Data"
	loadUserLibOnce sync.Once
	userLibLock     sync.Mutex
	usersLib        *UsersLib
	facesVectors    *FacesVectors = NewFacesVectors()
)

func NewFacesVectors() *FacesVectors {
	f := &FacesVectors{}
	f.Count = 0
	f.FacesVectors = make(map[string]*PersonVectors, 0)
	return f
}

func (f *FacesVectors) AddUser(firstname string, lastname string, faces [][]float64) *PersonVectors {
	user := &User{FirstName: firstname, LastName: lastname}
	faceVectors := eigenface.NewFaceVectors(100, 100, len(faces))
	faceVectors.Pixels = faces
	person := &PersonVectors{User: user, FaceVectors: faceVectors}
	f.FacesVectors[person.User.Key()] = person
	return person
}

func (f *FacesVectors) DetectFaces(imagepath string) [][]float64 {
	faces := make([][]float64, 0)
	userBasePath := DataPath + string(filepath.Separator) + "tmp"
	_, err := os.Stat(userBasePath)
	if err != nil {
		os.MkdirAll(userBasePath, os.ModePerm)
	}

	fd := facedetector.NewFaceDetector(imagepath)
	filespaths := fd.DrawImageInDirectory(userBasePath)
	for _, file := range filespaths {
		_, _, fv := ToVector(file)
		faces = append(faces, fv)
	}
	logger.Log("detected " + strconv.Itoa(len(faces)) + " faces.")
	return faces

}

func GetUsersLib() *UsersLib {
	loadUserLibOnce.Do(func() {
		usersLib = &UsersLib{}
		usersLib.UsersFace = make(map[string]*UserFace, 0)
		_, err := os.Stat(DataPath)
		if err != nil {
			os.MkdirAll(DataPath, os.ModePerm)
		}
		usersLib.load()
	})
	usersLib.save()
	return usersLib
}

func (ul *UsersLib) load() {
	f, err := os.Open(DataPath + string(filepath.Separator) + "data_library.json")
	if err != nil {
		logger.Log(err.Error())
		return
	}
	defer f.Close()
	err = json.NewDecoder(f).Decode(ul)
	if err != nil {
		logger.Log(err.Error())
		return
	}
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
		logger.Log(err.Error())
		return
	}
	defer f.Close()
	err = json.NewEncoder(f).Encode(u)
	if err != nil {
		logger.Log(err.Error())
		return
	}
}

func NewUserFace() *UserFace {
	return &UserFace{}
}

func (u *UserFace) GetKey() string {
	return u.User.FirstName + "." + u.User.LastName
}

func (u *UserFace) DetectFacesFromImages(images []image.Image) {
	userBasePath := DataPath + string(filepath.Separator) + u.GetKey()
	_, err := os.Stat(userBasePath)
	if err != nil {
		os.MkdirAll(userBasePath, os.ModePerm)
	}
	for _, img := range images {
		fd := facedetector.NewFaceDectectorFromImage(img)
		filespaths := fd.DrawImageInDirectory(userBasePath)
		u.TrainingImages = append(u.TrainingImages, filespaths...)
	}
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
	logger.Log("Found " + strconv.Itoa(len(u.TrainingImages)) + " faces.")
	return
}

func SumPixels(face []float64, width int, height int) float64 {
	sum := 0.
	for i := 0; i < (width * height); i++ {
		sum += face[i]
	}

	return math.Abs(sum / float64(width*height) / 0xffff)
}

func (u *UserFace) SaveAverageFace() {
	img := ToImage(u.AverageFace)
	out, err := os.Create(DataPath + string(filepath.Separator) + u.GetKey() + string(filepath.Separator) + "average.png")
	if err != nil {
		logger.Log(err.Error())
		return
	}

	err = png.Encode(out, img)
	if err != nil {
		logger.Log(err.Error())
		return
	}
}

func SaveImageTo(img *image.Gray16, path string) {
	out, err := os.Create(path)
	if err != nil {
		logger.Log(err.Error())
		return
	}
	defer out.Close()

	err = png.Encode(out, img)
	if err != nil {
		logger.Log(err.Error())
		return
	}
}

func (u *UserFace) SaveNormalizedFaces() {
	for index, face := range u.Faces {
		img := ToImage(face)
		out, err := os.Create(DataPath + string(filepath.Separator) + u.GetKey() + string(filepath.Separator) + strconv.Itoa(index) + "_normalized.png")
		if err != nil {
			logger.Log(err.Error())
			return
		}
		defer out.Close()

		err = png.Encode(out, img)
		if err != nil {
			logger.Log(err.Error())
			return
		}
	}
}

func (u *UserFace) TrainFaces() {
	var width int
	var height int
	if len(u.TrainingImages) == 0 {
		return
	}

	faces := make([][]float64, len(u.TrainingImages))
	for index, imagePath := range u.TrainingImages {
		var f []float64
		width, height, f = ToVector(imagePath)
		faces[index] = f
	}

	u.FacesDetected = faces
	u.Faces = eigenface.Normalize(faces, width, height)
	u.AverageFace = eigenface.Average(faces, width, height)
	u.SaveAverageFace()
	u.SaveNormalizedFaces()

	return
}

func (u *UsersLib) RecognizeFaceFromImage(image image.Image) *UserFace {

	sumConserved := 1.
	faceFound := ""
	faceFoundVector := &UserFace{}
	_, err := os.Stat("tmp")
	if err != nil {
		os.MkdirAll("tmp", os.ModePerm)
	}
	fd := facedetector.NewFaceDectectorFromImage(image)
	filespaths := fd.DrawImageInDirectory("tmp")
	facesDetected := make([][]float64, 0)
	for _, file := range filespaths {
		width, height, imageVector := ToVector(file)
		if height > 0 && width > 0 {
			facesDetected = append(facesDetected, imageVector)
		}
		for key, person := range u.UsersFace {
			logger.Log("Compare with " + key)
			if len(person.AverageFace) == 0 {
				continue
			}
			average := eigenface.Difference(person.AverageFace, imageVector, width, height)
			sum := SumPixels(average, width, height)

			if sum < sumConserved && sum < 0.05 {
				sumConserved = sum
				faceFound = key
				faceFoundVector = person
			}
			logger.Log(key + " : " + strconv.FormatFloat(sum, 'f', 10, 64) + " with " + file)
			//i := ToImage(average)
			//SaveImageTo(i, "tmp/average"+key+".png")
			//SaveImageTo(ToImage(person.AverageFace), "tmp/"+key+".png")
		}
	}
	logger.Log(faceFound + " seems to be the person you're looking for with value: " + strconv.FormatFloat(sumConserved, 'f', 10, 64))

	faceFoundVector.FacesDetected = facesDetected
	return faceFoundVector
}

func (u *UserFace) ComputeRecognizeDistance(face []float64) float64 {
	distance := 0.
	nbImages := len(u.Faces)
	nbPixels := len(u.Faces[0])
	for i := 0; i < nbPixels; i++ {
		tmp := 0.
		for j := 0; j < nbImages; j++ {
			tmp += u.Faces[j][i] - face[i]
		}
		distance += math.Pow(tmp/float64(nbPixels), 2)
	}

	return math.Sqrt(distance) / float64(nbImages)
}

func (u *UsersLib) RecognizeFace(imagePath string) *UserFace {

	sumConserved := 1.
	faceFound := ""
	faceFoundVector := &UserFace{}
	_, err := os.Stat("tmp")
	if err != nil {
		os.MkdirAll("tmp", os.ModePerm)
	}
	fd := facedetector.NewFaceDetector(imagePath)
	filespaths := fd.DrawImageInDirectory("tmp")
	for _, file := range filespaths {
		width, height, imageVector := ToVector(file)
		for key, person := range u.UsersFace {
			if len(person.AverageFace) == 0 {
				continue
			}
			average := eigenface.Difference(person.AverageFace, imageVector, width, height)
			sum := SumPixels(average, width, height)

			if sum < sumConserved && sum < 0.05 {
				sumConserved = sum
				faceFound = key
				faceFoundVector = person
			}
			logger.Log(key + " : " + strconv.FormatFloat(sum, 'f', 10, 64) + " with " + file)
			//i := ToImage(average)
			//SaveImageTo(i, "tmp/average"+key+".png")
			//SaveImageTo(ToImage(person.AverageFace), "tmp/"+key+".png")
		}
	}
	logger.Log(faceFound + " seems to be the person you're looking for with value: " + strconv.FormatFloat(sumConserved, 'f', 10, 64))
	//f := facedetector.NewFaceDetector(imagePath)
	//_, err := os.Stat("tmp")
	//if err != nil {
	//	os.MkdirAll("tmp", os.ModePerm)
	//}
	//filespaths := f.DrawImageInDirectory("tmp")
	//facesDetected := make([]eigenface.FaceVector, 0)
	//for _, filePath := range filespaths {
	//	fvector := ToVector(filePath)
	//	facesDetected = append(facesDetected, fvector)
	//}
	//averageFace := eigenface.Average(facesDetected)
	//
	//for key, userFace := range u.UsersFace {
	//
	//	result := eigenface.Difference(userFace.AverageFace, averageFace)
	//	logger.Log(strconv.Itoa(result.Height) + ":" + strconv.Itoa(result.Width))
	//	if result.Width == 0 && result.Height == 0 {
	//		logger.Log(imagePath + " contains " + key)
	//	}
	//
	//}
	return faceFoundVector
}
func (u *UsersLib) CompareFace(facepath string) {
	width, height, facevector := ToVector(facepath)
	facesVector := make([][]float64, 0)
	facesVector = append(facesVector, facevector)
	averageFace := eigenface.Average(facesVector, width, height)
	_, err := os.Stat("tmp")
	if err != nil {
		os.MkdirAll("tmp", os.ModePerm)
	}
	SaveImageTo(ToImage(averageFace), "tmp/face_temp.png")
	//for key, userFace := range u.UsersFace {
	//
	//	fvs := eigenface.NewFaceVectors(userFace.AverageFace.Height, userFace.AverageFace.Width, len(userFace.Faces))
	//	fvs.Pixels = userFace.Faces
	//	fvs.Train()
	//
	//	distance := fvs.ComputeDistance(facevector)
	//
	//	logger.Log(key + " Distance : " + strconv.FormatFloat(distance, 'b', -1, 64))
	//	//distance := eigenface.ComputeDistance(userFace.AverageFace.Pixels, averageFace.Pixels)
	//	////distance := userFace.LevenshteinDistance(averageFace)
	//	//logger.Log(key + " Distance : " + strconv.FormatFloat(distance, 'b', -1, 64))
	//	////faceVectorAv := eigenface.Average(faces)
	//	////logger.Log(key + " " + faceVectorAv.ToString())
	//	//diffVector := eigenface.Difference(userFace.AverageFace, averageFace)
	//	//distance = eigenface.ComputeDistance(diffVector.Pixels, userFace.AverageFace.Pixels)
	//	//logger.Log(key + " diff Distance : " + strconv.FormatFloat(distance, 'b', -1, 64))
	//}
}

func (u *UserFace) LevenshteinDistance(face1 []float64) float64 {
	s1 := len(face1)
	s2 := len(u.AverageFace)
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
			if face1[i] == u.AverageFace[j] {
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
