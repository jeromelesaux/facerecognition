package model

import (
	"encoding/json"
	"fmt"
	"github.com/jbuchbinder/gopnm"
	"github.com/jeromelesaux/facedetection/facedetector"
	"github.com/jeromelesaux/facerecognition/algorithm"
	"github.com/jeromelesaux/facerecognition/logger"
	"github.com/nfnt/resize"
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

type FaceRecognitionItem struct {
	User           User     `json:"user"`
	TrainingImages []string `json:"training_images"`
}

type FaceRecognitionLib struct {
	Items                  map[string]*FaceRecognitionItem `json:"facerecognition_lib"`
	MinimalNumOfComponents int
	Width                  int
	Height                 int
}

var (
	loadUserLibOnce sync.Once
	userLibLock     sync.Mutex
	lib             *FaceRecognitionLib
)

func NewFaceRecognitionLib() *FaceRecognitionLib {
	return &FaceRecognitionLib{Items: make(map[string]*FaceRecognitionItem, 0)}
}

func GetFaceRecognitionLib() *FaceRecognitionLib {

	loadUserLibOnce.Do(func() {
		lib = NewFaceRecognitionLib()
		_, err := os.Stat(GetConfig().GetDataLib())
		if err != nil {
			os.MkdirAll(GetConfig().GetDataLib(), os.ModePerm)
		}
		_, err = os.Stat(GetConfig().GetTmpDirectory())
		if err != nil {
			os.MkdirAll(GetConfig().GetTmpDirectory(), os.ModePerm)
		}
		lib.load()
	})
	lib.NormalizeImageLength()
	return lib
}

func (frl *FaceRecognitionLib) load() {
	f, err := os.Open(GetConfig().GetDataLib())
	if err != nil {
		logger.Log(err.Error())
		return
	}
	defer f.Close()
	err = json.NewDecoder(f).Decode(frl)
	if err != nil {
		logger.Log(err.Error())
		return
	}
	frl.MinimalNumOfComponents = len(frl.Items)
}

func (frl *FaceRecognitionLib) AddUserFace(u *FaceRecognitionItem) {
	frl.Items[u.GetKey()] = u
	if len(u.TrainingImages) > 0 && len(u.TrainingImages) < 4 {
		frl.MinimalNumOfComponents = len(u.TrainingImages)
	}
	frl.save()
}

func (frl *FaceRecognitionLib) save() {
	userLibLock.Lock()
	defer userLibLock.Unlock()
	f, err := os.Create(GetConfig().GetDataLib())
	if err != nil {
		logger.Log(err.Error())
		return
	}
	defer f.Close()
	err = json.NewEncoder(f).Encode(frl)
	if err != nil {
		logger.Log(err.Error())
		return
	}
}

func (frl *FaceRecognitionLib) NormalizeImageLength() {
	width := 100000
	height := 100000
	var wc sync.WaitGroup

	for _, user := range frl.Items {
		wc.Add(1)
		go func() {
			defer wc.Done()
			for _, img := range user.TrainingImages {
				logger.Log("Image loaded : " + img + " for user " + user.GetKey())
				f, err := os.Open(img)
				if err == nil {
					defer f.Close()
					i, _, err := image.Decode(f)
					if err == nil {
						imgWidth := i.Bounds().Max.X
						imgHeight := i.Bounds().Max.Y
						logger.Log("image size : " + strconv.Itoa(imgWidth) + "," + strconv.Itoa(imgHeight))
						if imgWidth < width {
							width = imgWidth
						}
						if imgHeight < height {
							height = imgHeight
						}
					}
				}
			}
		}()
	}
	wc.Wait()
	frl.Width = width
	frl.Height = width
	for _, user := range frl.Items {
		for _, img := range user.TrainingImages {
			normalizeImage(frl, img)
		}
	}
}

func normalizeImage(frl *FaceRecognitionLib, path string) {
	logger.Log("Normalized image " + path + " with size width " + strconv.Itoa(frl.Width) + " and height " + strconv.Itoa(frl.Height))
	f, err := os.Open(path)
	if err == nil {
		defer f.Close()
		i, _, err := image.Decode(f)
		if err == nil {
			ir := resize.Resize(uint(frl.Width), uint(frl.Height), i, resize.Lanczos3)
			fw, err := os.Create(path)
			if err == nil {
				defer fw.Close()
				err = pnm.Encode(fw, ir, pnm.PGM)
				if err != nil {
					logger.Log(err.Error())
				}
			} else {
				logger.Log(err.Error())
			}
		} else {
			logger.Log(err.Error())
		}
	}
}

func (frl *FaceRecognitionLib) MatrixNVectorize(img *image.Image) *algorithm.Matrix {
	filename := GetConfig().GetTmpDirectory() + "raw.pgm"
	f, err := os.Create(filename)
	if err != nil {
		logger.Log(err.Error())
		return &algorithm.Matrix{}
	}
	defer f.Close()
	pnm.Encode(f, *img, pnm.PGM)
	normalizeImage(frl, filename)
	return ToMatrix(filename).Vectorize()
}

func (frl *FaceRecognitionLib) FindFace(img *image.Image) ([]*algorithm.Matrix, []string) {
	mats := make([]*algorithm.Matrix, 0)
	filesnames := make([]string, 0)
	fd := facedetector.NewFaceDetector(*img, GetConfig().FaceDetectionConfigurationFile)
	var wc sync.WaitGroup

	for i, r := range fd.GetFaces() {
		wc.Add(1)
		go func() {
			logger.Logf("%v", r)
			defer wc.Done()
			b := make([]byte, 16)
			rand.Read(b)
			id := fmt.Sprintf("tofind-%X", b)
			dstRect := image.Rect(r.X, r.Y, (r.X + r.Width), (r.Y + r.Height))
			dst := image.NewRGBA(dstRect)
			draw.Draw(dst, dstRect, fd.Image, image.Point{r.X, r.Y}, draw.Src)
			filename := GetConfig().GetTmpDirectory() + "face_" + id + "_" + strconv.Itoa(r.X) + "_" + strconv.Itoa(r.Y) + "_" + strconv.Itoa(r.Width) + "_" + strconv.Itoa(r.Height) + strconv.Itoa(i) + ".png"
			fdst, _ := os.Create(filename)
			defer fdst.Close()
			png.Encode(fdst, dst)
			logger.Log("File " + filename + "saved as png.")
			newFilename := ToPgm(filename)
			os.Remove(filename)
			normalizeImage(frl, newFilename)
			mats = append(mats, ToMatrix(newFilename).Vectorize())
			filesnames = append(filesnames, newFilename)
		}()
	}
	wc.Wait()
	return mats, filesnames
}
func NewFaceRecognitionItem() *FaceRecognitionItem {
	return &FaceRecognitionItem{User: User{}}
}

func (fi *FaceRecognitionItem) GetKey() string {
	return fi.User.FirstName + "." + fi.User.LastName
}

func (fi *FaceRecognitionItem) DetectFacesFromImages(images []image.Image) {
	userBasePath := GetConfig().GetFaceRecognitionBasePath() + fi.GetKey()
	_, err := os.Stat(userBasePath)
	if err != nil {
		os.MkdirAll(userBasePath, os.ModePerm)
	}
	for _, img := range images {
		fd := facedetector.NewFaceDetector(img, GetConfig().FaceDetectionConfigurationFile)
		fi.storeImages(fd, userBasePath)
	}
}

func (fi *FaceRecognitionItem) storeImages(fd *facedetector.FaceDetector, basePath string) {
	rand.Seed(time.Now().UTC().UnixNano())
	var wc sync.WaitGroup

	for i, r := range fd.GetFaces() {
		wc.Add(1)
		go func() {
			defer wc.Done()
			b := make([]byte, 16)
			rand.Read(b)
			id := fmt.Sprintf("%s-%X", fi.User.Key(), b)
			dstRect := image.Rect(r.X, r.Y, (r.X + r.Width), (r.Y + r.Height))
			dst := image.NewRGBA(dstRect)
			draw.Draw(dst, dstRect, fd.Image, image.Point{r.X, r.Y}, draw.Src)
			filename := basePath + string(filepath.Separator) + "face_" + id + "_" + strconv.Itoa(r.X) + "_" + strconv.Itoa(r.Y) + "_" + strconv.Itoa(r.Width) + "_" + strconv.Itoa(r.Height) + strconv.Itoa(i) + ".png"
			fdst, _ := os.Create(filename)
			defer fdst.Close()
			png.Encode(fdst, dst)
			logger.Log("File " + filename + "saved as png.")
			newFilename := ToPgm(filename)
			fi.TrainingImages = append(fi.TrainingImages, newFilename)
			os.Remove(filename)
			logger.Log("File " + filename + " removed")
		}()
	}
	wc.Wait()
	return
}

func (fi *FaceRecognitionItem) DetectFaces(images []string) int {

	userBasePath := GetConfig().GetFaceRecognitionBasePath() + fi.GetKey()
	if _, err := os.Stat(userBasePath); os.IsNotExist(err) {
		os.MkdirAll(userBasePath, os.ModePerm)
	}
	var wc sync.WaitGroup

	for _, img := range images {
		wc.Add(1)
		go func() {
			defer wc.Done()
			logger.Log("Searching faces in image file : " + img)
			fd := facedetector.NewFaceDetector(img, GetConfig().FaceDetectionConfigurationFile)
			fi.storeImages(fd, userBasePath)
		}()
	}
	logger.Log("Found " + strconv.Itoa(len(fi.TrainingImages)) + " faces.")
	wc.Wait()
	return len(fi.TrainingImages)
}

func (frl *FaceRecognitionLib) ImportIntoDB(face *facedetector.FaceDetector, user *FaceRecognitionItem) *FaceRecognitionItem {
	basePath := GetConfig().GetFaceRecognitionBasePath() + user.GetKey() + string(filepath.Separator)
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		os.MkdirAll(basePath, os.ModePerm)
	}

	user.storeImages(face, basePath)
	frl.AddUserFace(user)
	return user
}

func (frl *FaceRecognitionLib) Train(modelType string) {
	t := frl.GetTrainer(modelType)
	t.Train()
}

func (frl *FaceRecognitionLib) GetTrainer(modelType string) *Trainer {
	// recuperation du nombre minimal d'image d'entrainement pour
	// determiner numOfComponents
	// et ne pas insérer l'image d'un utilisateur sir numOfComponents est
	// dépassé pour cet utilisateur.
	getDistanceFunc := &L1{}
	t := NewTrainerArgs(modelType, len(frl.Items), len(frl.Items)+1, getDistanceFunc.GetDistance)

	for username, user := range frl.Items {
		numOfComponents := 0
		if len(user.TrainingImages) > 0 {
			for _, path := range user.TrainingImages {
				numOfComponents++
				if numOfComponents > frl.MinimalNumOfComponents {
					break
				} else {
					t.Add(ToMatrix(path).Vectorize(), username)
				}
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
