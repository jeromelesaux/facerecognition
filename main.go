package main

import (
	"facerecognition/web"
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/train", web.Training)
	http.HandleFunc("/compare", web.Compare)
	http.Handle("/", http.StripPrefix("/", http.FileServer(http.Dir("./static"))))
	err := http.ListenAndServe(":8099", nil)
	if err != nil {
		fmt.Println(err)
	}
}
