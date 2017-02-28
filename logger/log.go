package logger

import (
	"fmt"
	"os"
	"time"
)

func Log(message string) {
	fmt.Fprintf(os.Stdout, "%s : %s\n", time.Now().Format(time.RFC3339), message)
}
