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
	"io/ioutil"
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
	TrainingImages []string `-`
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
	return &FaceRecognitionLib{
		Items: make(map[string]*FaceRecognitionItem, 0),
		MinimalNumOfComponents: 10,
		Width:  92,
		Height: 92,
	}
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
	if len(lib.Items) > 0 {
		lib.NormalizeImageLength()
	}
	return lib
}

func (fl *FaceRecognitionLib) load() {
	f, err := os.Open(GetConfig().GetDataLib())
	if err != nil {
		logger.Logf("cannot open datalib %s , error :%v", GetConfig().GetDataLib(), err.Error())
		return
	}
	defer f.Close()
	err = json.NewDecoder(f).Decode(fl)
	if err != nil {
		logger.Logf("cannot not decode configuration file %s, error :%v", GetConfig().GetDataLib(), err.Error())
		return
	}
	fl.loadItems()

	//frl.MinimalNumOfComponents = len(frl.Items)
}
func (fl *FaceRecognitionLib) loadItems() {
	for key, _ := range fl.Items {
		userDir := fl.Items[key].GetKey()
		directoryToScan := GetConfig().GetFaceRecognitionBasePath() + userDir
		fs, err := ioutil.ReadDir(directoryToScan)
		if err != nil {
			logger.Logf("error while scanning directory %s, with error :%v", directoryToScan, err)
			continue
		}

		diff := fl.MinimalNumOfComponents - len(fs)
		for i, file := range fs {
			if i >= fl.MinimalNumOfComponents {
				break
			}
			filePath := directoryToScan + separator + file.Name()
			fl.Items[key].TrainingImages = append(fl.Items[key].TrainingImages, filePath)
		}
		// to be compliant with the number of MinimalNumOfComponents
		if diff > 0 {
			i := diff
			for i > 0 {
				for j := len(fs) - 1; j >= 0 && i > 0; j-- {
					filePath := directoryToScan + separator + fs[j].Name()
					logger.Logf("extra adding to %s file %s", key, filePath)
					fl.Items[key].TrainingImages = append(fl.Items[key].TrainingImages, filePath)
					i--
				}
			}
		}
	}
}

func (fl *FaceRecognitionLib) AddUserFace(u *FaceRecognitionItem) {
	fl.Items[u.GetKey()] = u
	if len(u.TrainingImages) > 0 && len(u.TrainingImages) < 4 {
		fl.MinimalNumOfComponents = len(u.TrainingImages)
	}
	fl.save()
}

func (fl *FaceRecognitionLib) save() {
	userLibLock.Lock()
	defer userLibLock.Unlock()
	f, err := os.Create(GetConfig().GetDataLib())
	if err != nil {
		logger.Logf("cannot create datalib %s, error:%v", GetConfig().GetDataLib(), err.Error())
		return
	}
	defer f.Close()
	err = json.NewEncoder(f).Encode(fl)
	if err != nil {
		logger.Logf("cannot encode to json datalib file %s, with error %v", GetConfig().GetDataLib(), err.Error())
		return
	}
}

func (fl *FaceRecognitionLib) NormalizeImageLength() {
	width := 100000
	height := 100000
	var wc sync.WaitGroup

	for _, user := range fl.Items {
		wc.Add(1)
		go func() {
			defer wc.Done()
			for _, img := range user.TrainingImages {
				f, err := os.Open(img)
				if err == nil {
					defer f.Close()
					i, _, err := image.Decode(f)
					if err == nil {
						imgWidth := i.Bounds().Max.X
						imgHeight := i.Bounds().Max.Y
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
	for _, user := range fl.Items {
		for _, img := range user.TrainingImages {
			normalizeImage(fl, img)
		}
	}
}

func normalizeImage(fl *FaceRecognitionLib, path string) {
	f, err := os.Open(path)
	if err == nil {
		defer f.Close()
		i, _, err := image.Decode(f)
		if err == nil {
			ir := resize.Resize(uint(fl.Width), uint(fl.Height), i, resize.Lanczos3)
			fw, err := os.Create(path)
			if err == nil {
				defer fw.Close()
				err = pnm.Encode(fw, ir, pnm.PGM)
				if err != nil {
					logger.Logf("cannot encode to pnm file %s with error %v", path, err.Error())
				}
			} else {
				logger.Logf("cannot create file %s with error %v", path, err.Error())
			}
		} else {
			logger.Logf("cannot decode image file %s, with error :%v", path, err.Error())
		}
	}
}

func (fl *FaceRecognitionLib) MatrixNVectorize(img *image.Image) *algorithm.Matrix {
	filename := GetConfig().GetTmpDirectory() + "raw.pgm"
	f, err := os.Create(filename)
	if err != nil {
		logger.Logf("cannot create file %s with error %v", filename, err.Error())
		return &algorithm.Matrix{}
	}
	defer f.Close()
	pnm.Encode(f, *img, pnm.PGM)
	normalizeImage(fl, filename)
	return ToMatrix(filename).Vectorize()
}

func (fl *FaceRecognitionLib) FindFace(img *image.Image) ([]*algorithm.Matrix, []string) {
	mats := make([]*algorithm.Matrix, 0)
	filesnames := make([]string, 0)
	fd := facedetector.NewFaceDetector(*img, GetConfig().FaceDetectionConfigurationFile)
	var wc sync.WaitGroup

	for i, r := range fd.GetFaces() {
		wc.Add(1)
		go func() {
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
			normalizeImage(fl, newFilename)
			mats = append(mats, ToMatrix(newFilename).Vectorize())
			filesnames = append(filesnames, newFilename)
		}()
	}

	filename := GetConfig().GetTmpDirectory() + "final-faces-found.png"
	fdst, _ := os.Create(filename)
	defer fdst.Close()
	if err := png.Encode(fdst, fd.DrawFaces()); err != nil {
		logger.Logf("cannot encode to png file %s with error : %v", filename, err)
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

func (fl *FaceRecognitionLib) ImportIntoDB(face *facedetector.FaceDetector, user *FaceRecognitionItem) *FaceRecognitionItem {
	basePath := GetConfig().GetFaceRecognitionBasePath() + user.GetKey() + string(filepath.Separator)
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		os.MkdirAll(basePath, os.ModePerm)
	}
	user.storeImages(face, basePath)
	fl.AddUserFace(user)
	return user
}

func (fl *FaceRecognitionLib) Train(featureType string) {
	t := fl.GetTrainer(featureType)
	t.Train()
}

func (fl *FaceRecognitionLib) GetTrainer(featureType string) *Trainer {
	// recuperation du nombre minimal d'image d'entrainement pour
	// determiner numOfComponents
	// et ne pas insérer l'image d'un utilisateur sir numOfComponents est
	// dépassé pour cet utilisateur.
	// K's choice explained here http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
	getDistanceFunc := &L1{}
	t := NewTrainerArgs(featureType, 2, len(fl.Items)+1, getDistanceFunc.GetDistance)

	for username, user := range fl.Items {
		numOfComponents := 0
		if len(user.TrainingImages) > 0 {
			for _, path := range user.TrainingImages {
				numOfComponents++
				if numOfComponents > fl.MinimalNumOfComponents {
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
		logger.Logf("cannot encode to png file %s, with error %v", path, err.Error())
		return
	}
}
