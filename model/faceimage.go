package model

import (
	"facerecognition/eigen"
	"fmt"
	"github.com/disintegration/imaging"
	"image"
	"image/color"
	_ "image/png"
	"os"
)

var (
	Height = 100
	Width  = 100
)

func ToVector(path string) eigenface.FaceVector {
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(err.Error())
		return eigenface.FaceVector{}
	}
	defer f.Close()
	i, _, _ := image.Decode(f)
	i = Resize(i)
	width := i.Bounds().Max.X - i.Bounds().Min.X
	height := i.Bounds().Max.Y - i.Bounds().Min.Y
	minX := i.Bounds().Min.X
	minY := i.Bounds().Min.Y

	face := eigenface.FaceVector{Height: Height, Width: Width}
	face.Pixels = make([]float64, face.Width*face.Height)

	// iterate through image row by row
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			color := i.At(x-minX, y-minY)
			// ORL database images are 16-bit grayscale, so can use any of RGB values
			value, _, _, _ := color.RGBA()
			face.Pixels[(y*width)+x] = float64(value)
		}
	}
	return face
}

func ToImage(face eigenface.FaceVector) *image.Gray16 {
	bounds := image.Rect(0, 0, Width, Height)
	im := image.NewGray16(bounds)
	for y := 0; y < Height; y++ {
		for x := 0; x < Width; x++ {
			// ORL database images are 16-bit grayscale
			value := uint16(face.Pixels[(y*Width)+x])
			im.SetGray16(x, y, color.Gray16{value})
		}
	}
	return im
}

func Resize(img image.Image) *image.NRGBA {
	return imaging.Resize(img, Width, Height, imaging.Lanczos)
}
