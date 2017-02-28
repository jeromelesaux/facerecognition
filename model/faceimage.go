package model

import (
	"facerecognition/algorithm"
	"facerecognition/logger"
	"github.com/disintegration/imaging"
	_ "github.com/jbuchbinder/gopnm"
	"image"
	"image/color"
	_ "image/png"
	"os"
)

var (
	Height = 100
	Width  = 100
)

func StreamToVector(img image.Image) []float64 {
	i := Resize(img)
	width := i.Bounds().Max.X - i.Bounds().Min.X
	height := i.Bounds().Max.Y - i.Bounds().Min.Y
	minX := i.Bounds().Min.X
	minY := i.Bounds().Min.Y

	face := make([]float64, width*height)

	// iterate through image row by row
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			color := i.At(x-minX, y-minY)
			// ORL database images are 16-bit grayscale, so can use any of RGB values
			value, _, _, _ := color.RGBA()
			face[y+x] = float64(value)
		}
	}
	return face
}

func ToVector(path string) (int, int, []float64) {
	f, err := os.Open(path)
	if err != nil {
		logger.Log(err.Error())
		return 0, 0, make([]float64, 0)
	}
	defer f.Close()
	i, _, _ := image.Decode(f)
	i = Resize(i)
	width := i.Bounds().Max.X - i.Bounds().Min.X
	height := i.Bounds().Max.Y - i.Bounds().Min.Y
	minX := i.Bounds().Min.X
	minY := i.Bounds().Min.Y

	face := make([]float64, width*height)

	// iterate through image row by row
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			color := i.At(x-minX, y-minY)
			// ORL database images are 16-bit grayscale, so can use any of RGB values
			value, _, _, _ := color.RGBA()
			face[y+x] = float64(value)
		}
	}
	return width, height, face
}

func ToImage(face []float64) *image.Gray16 {
	bounds := image.Rect(0, 0, Width, Height)
	im := image.NewGray16(bounds)
	for y := 0; y < Height; y++ {
		for x := 0; x < Width; x++ {
			// ORL database images are 16-bit grayscale
			value := uint16(face[y+x])
			im.SetGray16(x, y, color.Gray16{value})
		}
	}
	return im
}

func Resize(img image.Image) *image.NRGBA {
	return imaging.Resize(img, Width, Height, imaging.Lanczos)
}

func ToMatrix(width, height int, face []float64) *algorithm.Matrix {
	m := algorithm.NewMatrix(height, width)
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			m.A[row][col] = face[row+col]
		}
	}
	return m
}
