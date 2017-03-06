package model

import (
	"encoding/json"
	"facedetection/facedetector"
	"facerecognition/logger"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
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
	User           User     `json:"user"`
	TrainingImages []string `json:"training_images"`
}

type UsersLib struct {
	UsersFace              map[string]*UserFace `json:"users_lib"`
	MinimalNumOfComponents int
}

var (
	DataPath        = "Data"
	loadUserLibOnce sync.Once
	userLibLock     sync.Mutex
	usersLib        *UsersLib
	//facesVectors    *FacesVectors = NewFacesVectors()
)

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
	if len(u.TrainingImages) > 0 && len(u.TrainingImages) < 4 {
		ul.MinimalNumOfComponents = len(u.TrainingImages)
	}
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
		rand.Seed(time.Now().UTC().UnixNano())
		for i, r := range fd.GetFaces() {
			b := make([]byte, 16)
			rand.Read(b)
			id := fmt.Sprintf("%s-%X", u.User.Key(), b)
			dstRect := image.Rect(r.X, r.Y, (r.X + r.Width), (r.Y + r.Height))
			dst := image.NewRGBA(dstRect)
			draw.Draw(dst, dstRect, fd.Image, image.Point{r.X, r.Y}, draw.Src)
			filename := DataPath + "face_" + id + "_" + strconv.Itoa(r.X) + "_" + strconv.Itoa(r.Y) + "_" + strconv.Itoa(r.Width) + "_" + strconv.Itoa(r.Height) + strconv.Itoa(i) + ".png"
			fdst, _ := os.Create(filename)
			defer fdst.Close()
			png.Encode(fdst, dst)
			logger.Log("File " + filename + "saved as png.")
			newFilename := ToPgm(filename)
			u.TrainingImages = append(u.TrainingImages, newFilename)
			os.Remove(filename)
			logger.Log("File " + filename + " removed")
		}
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

func (ul *UsersLib) ImportIntoDB(face *facedetector.FaceDetector, user *UserFace) *UserFace {
	basePath := DataPath + string(filepath.Separator) + user.GetKey() + string(filepath.Separator)
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		os.MkdirAll(basePath, os.ModePerm)
	}
	rand.Seed(time.Now().UTC().UnixNano())
	for i, r := range face.GetFaces() {
		b := make([]byte, 16)
		rand.Read(b)
		id := fmt.Sprintf("%s-%X", user.User.Key(), b)
		dstRect := image.Rect(r.X, r.Y, (r.X + r.Width), (r.Y + r.Height))
		dst := image.NewRGBA(dstRect)
		draw.Draw(dst, dstRect, face.Image, image.Point{r.X, r.Y}, draw.Src)
		filename := basePath + "face_" + id + "_" + strconv.Itoa(r.X) + "_" + strconv.Itoa(r.Y) + "_" + strconv.Itoa(r.Width) + "_" + strconv.Itoa(r.Height) + strconv.Itoa(i) + ".png"
		fdst, _ := os.Create(filename)
		defer fdst.Close()
		png.Encode(fdst, dst)
		logger.Log("File " + filename + "saved as png.")
		newFilename := ToPgm(filename)
		user.TrainingImages = append(user.TrainingImages, newFilename)
		os.Remove(filename)
		logger.Log("File " + filename + " removed")
	}
	ul.AddUserFace(user)
	return user
}

func (ul *UsersLib) GetTrainer() *Trainer {
	// recuperation du nombre minimal d'image d'entrainement pour
	// determiner numOfComponents
	// et ne pas insérer l'image d'un utilisateur sir numOfComponents est
	// dépassé pour cet utilisateur.
	getDistanceFunc := &CosineDissimilarity{}
	t := NewTrainerArgs("PCA", 1, ul.MinimalNumOfComponents, getDistanceFunc.GetDistance)

	for username, user := range ul.UsersFace {
		numOfComponents := 0
		for _, path := range user.TrainingImages {
			numOfComponents++
			if numOfComponents > ul.MinimalNumOfComponents {
				break
			} else {
				t.Add(ToMatrix(path).Vectorize(), username)
			}
		}
	}
	return t
}

func SumPixels(face []float64, width int, height int) float64 {
	sum := 0.
	for i := 0; i < (width * height); i++ {
		sum += face[i]
	}

	return math.Abs(sum / float64(width*height) / 0xffff)
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
