package web

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"facerecognition/eigen"
	"facerecognition/model"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"io"
	"mime/multipart"
	"net/http"
)

type FaceRecognitionResponse struct {
	Error            string     `json:"error,ompitempty"`
	User             model.User `json:"user"`
	Average          string     `json:"average"`
	FaceDetected     []string   `json:"faces_detected"`
	PersonRecognized string     `json:"person_recognized"`
}

func Compare(w http.ResponseWriter, r *http.Request) {
	var err error
	user := &model.User{}
	response := &FaceRecognitionResponse{PersonRecognized: "Not recognized"}
	userslib := model.GetUsersLib()

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

			fmt.Println(part.FileName())
			img, err := imageFromMultipart(part)
			if err == nil {
				faceFound := userslib.RecognizeFaceFromImage(img)
				if faceFound.User.LastName != "" && faceFound.User.FirstName != "" {
					user.FirstName = faceFound.User.FirstName
					user.LastName = faceFound.User.LastName
					response.FaceDetected = make([]string, 0)
					for _, f := range faceFound.FacesDetected {
						response.FaceDetected = append(response.FaceDetected, faceVectorToBase64(f))
					}
					response.User = *user
					response.Average = faceVectorToBase64(faceFound.AverageFace)
					response.PersonRecognized = "It seems to be " + faceFound.GetKey()
				}
			}
		}
	}

}

func Training(w http.ResponseWriter, r *http.Request) {
	var err error
	user := &model.User{}
	response := &FaceRecognitionResponse{}
	userslib := model.GetUsersLib()
	userFace := model.NewUserFace()
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
				fmt.Println(part.FileName())
				img, err := imageFromMultipart(part)
				if err == nil {
					images = append(images, img)
				}

			}

		}
	}
	fmt.Println(user)
	if user.FirstName == "" || user.LastName == "" {
		response.Error = "Firstname and lastname are mandatories."
	}
	if len(images) == 0 {
		response.Error = "No images detected"
	} else {
		userFace.User = *user
		userFace.DetectFacesFromImages(images)
		userFace.TrainFaces()
		userslib.AddUserFace(userFace)
		response.Average = faceVectorToBase64(userFace.AverageFace)
	}

	response.User = *user
	return
}

func sendJson(w http.ResponseWriter, i interface{}) {
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(i)
	if err != nil {
		fmt.Println(err)
	}
}

func faceVectorToBase64(f eigenface.FaceVector) string {
	img := model.ToImage(f)
	buf := new(bytes.Buffer)
	err := png.Encode(buf, img)
	if err != nil {
		fmt.Println(err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func imageFromMultipart(p *multipart.Part) (image.Image, error) {
	buf := bufio.NewReader(p)
	image, _, err := image.Decode(buf)
	if err != nil {
		fmt.Println(err)
	}
	return image, err
}

func stringFromMultipart(p *multipart.Part) string {
	buf := new(bytes.Buffer)
	buf.ReadFrom(p)
	return buf.String()
}
