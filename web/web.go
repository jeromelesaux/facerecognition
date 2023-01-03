package web

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"sync"

	"github.com/jeromelesaux/facerecognition/algorithm"
	"github.com/jeromelesaux/facerecognition/logger"
	"github.com/jeromelesaux/facerecognition/model"
)

type FaceRecognitionResponse struct {
	Error            string     `json:"error,omitempty"`
	User             model.User `json:"user"`
	Average          string     `json:"average"`
	FaceDetected     []string   `json:"faces_detected"`
	PersonRecognized string     `json:"person_recognized"`
	Distance         float64    `json:"distance"`
}

type PersonResponse struct {
	FirstName string   `json:"first_name"`
	LastName  string   `json:"last_name"`
	Faces     []string `json:"faces"`
}

type LibraryResponse struct {
	Error   string           `json:"error,omitempty"`
	Persons []PersonResponse `json:"persons"`
}

func NewLibraryResponse() *LibraryResponse {
	return &LibraryResponse{Persons: make([]PersonResponse, 0)}
}

var (
	t       *model.Trainer
	frlib   *model.FaceRecognitionLib
	libload sync.Once
)

func load() {
	libload.Do(func() {
		frlib = model.GetFaceRecognitionLib()
		t = frlib.GetTrainer("PCA")
		t.Train()
	})
}

func GetPerson(w http.ResponseWriter, r *http.Request) {
	load()
	key, ok := r.URL.Query()["id"]
	if !ok {
		w.WriteHeader(404)
		sendJson(w, "not found")
	}
	for _, v := range frlib.Items {
		if v.GetKey() == key[0] {
			p := PersonResponse{FirstName: v.User.FirstName, LastName: v.User.LastName}
			for _, f := range v.TrainingImages {
				p.Faces = append(p.Faces, fileToBase64(f))
			}
			w.WriteHeader(200)
			sendJson(w, p)
			return
		}
	}
	w.WriteHeader(404)
	sendJson(w, "not found")
}

func ListPersons(w http.ResponseWriter, r *http.Request) {
	load()
	response := NewLibraryResponse()

	defer func() {
		w.WriteHeader(200)
		sendJson(w, response)
	}()

	for _, v := range frlib.Items {
		p := PersonResponse{FirstName: v.User.FirstName, LastName: v.User.LastName}

		/*for _,f := range v.TrainingImages {
			p.Faces = append(p.Faces,fileToBase64(f))
		}*/
		response.Persons = append(response.Persons, p)
	}
}

func Compare(w http.ResponseWriter, r *http.Request) {
	load()
	var err error
	// user := &model.User{}
	response := &FaceRecognitionResponse{PersonRecognized: "Not recognized"}

	defer func() {
		w.WriteHeader(200)
		sendJson(w, response)
	}()

	mr, err := r.MultipartReader()
	if err != nil {
		response.Error = err.Error()
		return
	}
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			response.Error = err.Error()
			return
		}
		if name := part.FormName(); name != "" {
			logger.Log(part.FileName())
			img, err := imageFromMultipart(part)
			if err == nil {
				mats, files := frlib.FindFace(&img)
				frlib.Train(model.PCAFeatureType)
				if len(mats) == 0 {
					mats = append(mats, frlib.MatrixNVectorize(&img))
				}
				for _, m := range mats {
					p, distance := t.Recognize(m)
					logger.Log("Found " + p + " distance " + strconv.FormatFloat(distance, 'e', 2, 32))
					if p != "" {
						response.User = frlib.Items[p].User
						response.Distance = 0.0
						if len(files) == 0 {
							response.Average = imageToBase64(&img)
						} else {
							response.Average = fileToBase64(model.GetConfig().GetTmpDirectory() + "final-faces-found.png")
						}
						for _, f := range frlib.Items[p].TrainingImages {
							response.FaceDetected = append(response.FaceDetected, fileToBase64(f))
						}
						response.PersonRecognized = "It seems to be " + response.User.ToString()
					} else {
						response.Error = "Not recognized."
					}
				}
			}
		}
	}
}

func Training(w http.ResponseWriter, r *http.Request) {
	load()
	var err error
	user := &model.User{}
	response := &FaceRecognitionResponse{}
	frlib := model.GetFaceRecognitionLib()
	userFace := model.NewFaceRecognitionItem()
	images := make([]image.Image, 0)

	defer func() {
		w.WriteHeader(200)
		sendJson(w, response)
	}()

	mr, err := r.MultipartReader()
	if err != nil {
		response.Error = err.Error()
		return
	}

	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			response.Error = err.Error()
			return
		}
		if name := part.FormName(); name != "" {
			switch name {
			case "first_name":
				user.FirstName = stringFromMultipart(part)
				continue
			case "last_name":
				user.LastName = stringFromMultipart(part)
				continue
			default:
				logger.Log(part.FileName())
				img, err := imageFromMultipart(part)
				if err == nil {
					images = append(images, img)
				}

			}
		}
	}
	logger.Log(user.Key())
	if user.FirstName == "" || user.LastName == "" {
		response.Error = "Firstname and lastname are mandatories."
	}
	if len(images) == 0 {
		response.Error = "No images detected"
	} else {

		userFace.User = *user
		logger.Log("Adding " + userFace.GetKey())
		userFace.DetectFacesFromImages(images)
		frlib.AddUserFace(userFace)
	}

	response.User = *user
}

func sendJson(w http.ResponseWriter, i interface{}) {
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(i)
	if err != nil {
		logger.Log(err.Error())
	}
}

func faceVectorToBase64(f *algorithm.Matrix) string {
	img := model.ToImage(f)
	buf := new(bytes.Buffer)
	err := png.Encode(buf, img)
	if err != nil {
		logger.Log(err.Error())
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func fileToBase64(f string) string {
	fh, _ := os.Open(f)
	defer fh.Close()
	img, _, err := image.Decode(fh)
	if err != nil {
		logger.Log(err.Error())
	}

	buf := new(bytes.Buffer)
	err = png.Encode(buf, img)
	if err != nil {
		logger.Log(err.Error())
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func imageToBase64(img *image.Image) string {
	buf := new(bytes.Buffer)
	err := png.Encode(buf, *img)
	if err != nil {
		logger.Log(err.Error())
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func imageFromMultipart(p *multipart.Part) (image.Image, error) {
	buf := bufio.NewReader(p)
	image, _, err := image.Decode(buf)
	if err != nil {
		logger.Log(err.Error())
	}
	return image, err
}

func stringFromMultipart(p *multipart.Part) string {
	buf := new(bytes.Buffer)
	buf.ReadFrom(p)
	return buf.String()
}
