package logger

import (
	"testing"
)

func TestLogf(t *testing.T) {
	Logf("%d", 1)
	Logf("%d %s", 1, "hello")
}
