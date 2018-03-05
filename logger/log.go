package logger

import (
	"fmt"
	"os"
	"time"
)

func Log(message string) {
	fmt.Fprintf(os.Stdout, "%s : %s\n", time.Now().Format(time.RFC3339), message)
}

func Logf(format string, params ...interface{}) {

	logformat := "%s : " + format + "\n"
	parameters := make([]interface{}, len(params)+1)
	parameters[0] = time.Now().Format(time.RFC3339)
	for i := range params {
		parameters[i+1] = params[i]
	}
	//fmt.Printf("format is '%s'\n", logformat)
	//fmt.Printf("params :'%v'\n", parameters)
	fmt.Fprintf(os.Stdout, logformat, parameters...)
}
