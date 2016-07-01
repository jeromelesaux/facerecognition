package main

import (
	"facerecognition/web"
	"net/http"
)

func main() {
	http.HandleFunc("/train", web.Training)
}
