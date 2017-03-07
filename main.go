package main

import (
	"facerecognition/logger"
	"facerecognition/model"
	"facerecognition/web"
	"flag"
	"net/http"
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

var httpport = flag.String("httpport", "", "HTTP port value (default 8099)")
var firstname = flag.String("firstname", "", "Firstname of the person to add")
var lastname = flag.String("lastname", "", "Lastname ot the person to add")
var add = flag.Bool("add", false, "Add the person in user lib")

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
		if *add != false {

			lib := model.GetUsersLib()
			uf := model.NewUserFace()
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
