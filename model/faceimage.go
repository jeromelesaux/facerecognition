package model

import (
	"bufio"
	"facerecognition/algorithm"
	"facerecognition/logger"
	"github.com/disintegration/imaging"
	"github.com/jbuchbinder/gopnm"
	"image"
	"image/color"
	_ "image/png"
	"os"
	"strconv"
	"strings"
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
	//i = Resize(i)
	width := i.Bounds().Max.X - i.Bounds().Min.X
	height := i.Bounds().Max.Y - i.Bounds().Min.Y
	minX := i.Bounds().Min.X
	minY := i.Bounds().Min.Y

	face := make([]float64, width*height)

	// iterate through image row by row
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := i.At(x-minX, y-minY)
			// ORL database images are 16-bit grayscale, so can use any of RGB values
			r, g, b, _ := c.RGBA()
			grayValue := (19595*r + 38470*g + 7471*b + 1<<15) >> 24
			face[y+x] = float64(uint8(grayValue))
		}
	}
	return width, height, face
}

func ToImage(face *algorithm.Matrix) *image.Gray16 {
	bounds := image.Rect(0, 0, face.N, face.M)
	im := image.NewGray16(bounds)
	for y := 0; y < face.M; y++ {
		for x := 0; x < face.N; x++ {
			// ORL database images are 16-bit grayscale
			value := uint16(face.A[y][x])
			im.SetGray16(x, y, color.Gray16{value})
		}
	}
	return im
}

func Resize(img image.Image) *image.NRGBA {
	return imaging.Resize(img, Width, Height, imaging.Lanczos)
}

//
//func ToMatrix(width, height int, face []float64) *algorithm.Matrix {
//	m := height
//	n := width
//	fmt.Println(face)
//	mat := algorithm.NewMatrix(n * m,1)
//	for p:= 0; p < n; p++ {
//		for  q := 0; q < m; q++ {
//			mat.A[p*m +q][0] = face[q*p]
//		}
//	}
//	fmt.Println(mat)
//	return mat
//}
func ToMatrix(path string) *algorithm.Matrix {
	f, err := os.Open(path)
	if err != nil {
		logger.Log(err.Error())
		return algorithm.NewMatrix(0, 0)
	}
	defer f.Close()
	if strings.HasSuffix(path, ".pgm") {
		bf := bufio.NewReader(f)
		bf.ReadLine()
		d, _, _ := bf.ReadLine()
		dimensions := string(d[:])
		s := strings.Split(dimensions, " ")
		width, _ := strconv.Atoi(s[0])
		height, _ := strconv.Atoi(s[1])
		//fmt.Printf("%d %d", width, height)
		mat := algorithm.NewMatrix(height, width)
		bf.ReadLine()
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				value, _ := bf.ReadByte()
				mat.A[row][col] = float64(value)
			}
		}
		return mat
	} else {

		i, _, _ := image.Decode(f)
		//i = Resize(i)
		width := i.Bounds().Max.X - i.Bounds().Min.X
		height := i.Bounds().Max.Y - i.Bounds().Min.Y
		minX := i.Bounds().Min.X
		minY := i.Bounds().Min.Y
		matrix := algorithm.NewMatrix(height, width)

		// iterate through image row by row
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				c := i.At(x-minX, y-minY)
				// ORL database images are 16-bit grayscale, so can use any of RGB values
				pixel := color.GrayModel.Convert(c)
				r, g, b, _ := pixel.RGBA()

				//grayValue := (19595*r + 38470*g + 7471*b + 1<<15) >> 24
				grayValue := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
				matrix.A[y][x] = float64(uint8(grayValue / 256))
			}
		}

		return matrix
	}
}

func ToPgm(path string) string {
	index := strings.LastIndex(path, ".")
	pgmPath := path[0:index] + ".pgm"
	f, err := os.Create(pgmPath)
	if err != nil {
		logger.Log(err.Error())
		return ""
	}
	defer f.Close()
	f2, err := os.Open(path)
	if err != nil {
		logger.Log(err.Error())
		return ""
	}
	defer f2.Close()
	imgSrc, _, _ := image.Decode(f2)
	err = pnm.Encode(f, imgSrc, pnm.PGM)
	logger.Log("File : " + pgmPath + " created.")
	return pgmPath

}
