package main

import (
	"flag"
	"github.com/jeromelesaux/facerecognition/logger"
	"github.com/jeromelesaux/facerecognition/model"
	"github.com/jeromelesaux/facerecognition/web"
	"image"
	"net/http"
	"os"
	"strconv"
)

type stringflags []string

func (i *stringflags) String() string {
	return ""
}
func (i *stringflags) Set(value string) error {
	*i = append(*i, value)
	return nil
}

var imagesfiles stringflags

var httpport = flag.String("httpport", "", "HTTP port value (default 8099).")
var firstname = flag.String("firstname", "", "Firstname of the person to add.")
var lastname = flag.String("lastname", "", "Lastname ot the person to add.")
var add = flag.Bool("add", false, "Add the person in user lib.")
var recognize = flag.Bool("recognize", false, "Recognize person from image.")
var config = flag.String("config", "", "Path to the configuration file.")

type Value interface {
	String() string
	Set(string) error
}

func main() {

	flag.Var(&imagesfiles, "imagesfiles", "List of the images files of the person to add in database")
	flag.Parse()

	if flag.NFlag() == 0 {
		flag.PrintDefaults()
	} else {
		logger.Logf("configuration file %s", *config)
		if *config != "" {
			model.SetAndLoad(*config)
		} else {
			logger.Log("No configuration file set cannot continue.")
			return
		}
		if *recognize != false {
			lib := model.GetFaceRecognitionLib()
			t := lib.GetTrainer(model.PCAFeatureType)
			t.Train()
			for _, i := range imagesfiles {
				f, err := os.Open(i)
				if err != nil {
					logger.Logf("error while opening file %s with error %v", i, err)
				} else {
					defer f.Close()
					img, _, err := image.Decode(f)
					if err != nil {
						logger.Logf("error while decoding file %s with error %v", i, err)
					} else {
						mats, _ := lib.FindFace(&img)
						if len(mats) == 0 {
							mats = append(mats, lib.MatrixNVectorize(&img))
						}
						logger.Logf("found %d faces.", len(mats))
						for _, m := range mats {
							p, distance := t.Recognize(m)
							logger.Log("Found " + p + " distance " + strconv.FormatFloat(distance, 'e', 2, 32))
						}
					}
				}
			}
		} else {
			if *add != false {

				lib := model.GetFaceRecognitionLib()
				uf := model.NewFaceRecognitionItem()
				uf.User.FirstName = *firstname
				uf.User.LastName = *lastname
				logger.Log("Adding " + uf.GetKey())
				uf.DetectFaces(imagesfiles)
				lib.AddUserFace(uf)

			} else {
				if *httpport != "" {
					http.HandleFunc("/train", web.Training)
					http.HandleFunc("/compare", web.Compare)
					http.Handle("/", http.StripPrefix("/", http.FileServer(http.Dir("./static"))))
					err := http.ListenAndServe(":"+*httpport, nil)
					if err != nil {
						logger.Log(err.Error())
					}
				}
			}
		}
	}
}
