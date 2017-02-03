package eigenface

import (
	"github.com/gonum/matrix/mat64"
	"math"
	"strconv"
	"fmt"
)

type FaceVector struct {
	Pixels        []float64
	Width, Height int
}

type FaceVectors struct {
	Height int
	Width  int
	Pixels []FaceVector
	Mean   FaceVector
	Weight FaceVector
	Diff   []FaceVector
	Cov    []float64
	Eigen  [][]float64
}

func makeArray(length int) []float64 {
	return make([]float64, length)
}

func NewFaceVectors(height, width int, nbImage int) *FaceVectors {
	facevectors := &FaceVectors{Height: height, Width: width}
	facevectors.Pixels = make([]FaceVector, nbImage)
	facevectors.Diff = make([]FaceVector, nbImage)
	facevectors.Weight = FaceVector{Height: height, Width: width}
	facevectors.Weight.Pixels = makeArray(height * width)
	facevectors.Mean = FaceVector{Height: height, Width: width}
	facevectors.Mean.Pixels = makeArray(height * width)
	return facevectors
}

func (f *FaceVector) ToString() string {
	return "width:" + strconv.Itoa(f.Width) + ",Height:" + strconv.Itoa(f.Height)
}

func (f *FaceVectors) Train() {
	f.ComputeMean()
	f.ComputeDifferenceMatrixPixels()
	f.ComputeCovarianceMatrix()

	f.ComputeEigenFaces()
	f.ComputeWeights(f.Mean)
}

func (f *FaceVectors) ComputeMean() {
	for nbimages := 0; nbimages < len(f.Pixels); nbimages++ {
		sum := 0.
		for nbpixel := 0; nbpixel < (f.Width * f.Height); nbpixel++ {
			sum += f.Pixels[nbimages].Pixels[nbpixel]
			//f.Mean.Pixels[nbpixel] += f.Pixels[nbimages].Pixels[nbpixel]
		}
		//f.Mean.Pixels[nbpixel] /= float64(len(f.Pixels))
		f.Mean.Pixels[nbimages] = sum / float64(len(f.Pixels))
	}
	return
}

func (f *FaceVectors) ComputeDifferenceMatrixPixels() {

	for nbimages := 0; nbimages < len(f.Pixels); nbimages++ {
		f.Diff[nbimages].Pixels = makeArray(f.Width * f.Height)
		for nbpixels := 0; nbpixels < (f.Width * f.Height); nbpixels++ {
			f.Diff[nbimages].Pixels[nbpixels] = f.Pixels[nbimages].Pixels[nbpixels] - f.Mean.Pixels[nbimages]
		}
	}
	return
}

func (f *FaceVectors) ComputeCovarianceMatrix() {
	f.Cov = make([]float64, (f.Width*f.Height)*(f.Width*f.Height))

	for i := 0; i < (f.Width * f.Height); i++ {
		//f.Cov[i] = make([]float64, int64(f.Width*f.Height))
		for j := 0; j < (f.Width * f.Height); j++ {
			sum := 0.0
			for k := 0; k < len(f.Diff); k++ {
				sum += f.Diff[k].Pixels[i] * f.Diff[k].Pixels[j]
			}
			f.Cov[i+j] = sum
		}
	}
	//
	//for nbimages1 := 0; nbimages1 < len(f.Diff); nbimages1++ {
	//	f.Cov[nbimages1] = make([]float64,len(f.Diff))
	//	for nbimages2 := 0; nbimages2 < len(f.Diff); nbimages2++ {
	//		sum := 0.0
	//		for nbpixels := 0; nbpixels < len(f.Diff); nbpixels++ {
	//			sum += f.Diff[nbimages1].Pixels[nbpixels] * f.Diff[nbimages2].Pixels[nbpixels]
	//		}
	//		f.Cov[nbimages1][nbimages2] = sum
	//	}
	//}
	return
}

func (f *FaceVectors) ComputeEigenFaces() {
	//epsilon:= math.Pow(2, -52.0)

	eigenMatrix := mat64.Eigen{}


	denseMatrix := mat64.NewDense((f.Width*f.Height),(f.Width*f.Height),f.Cov)
	//mat64.Eigen(mat64.DenseCopyOf(denseMatrix),)
	d,_ := denseMatrix.Dims()
	fmt.Println(d)

	eigenMatrix.Factorize(denseMatrix,true)
	//denseMatrix := matrix.mat(f.Cov)
	//var es mat64.EigenSym
	//ok := es.Factorize(denseMatrix,true)
	////eigenVectors := mat64.Eigen{}
	////ok := eigenVectors.Factorize(denseMatrix,true)
	//if !ok {
	//	return
	//}
	////

	//eigenMatrix := mat64.Eigen(mat64.DenseCopyOf(denseMatrix),epsilon)
	//
	////
	////if err != nil {
	////	fmt.Println(err.Error())
	////	return
	////}
	////eigenVectors := es
	//
	eigenValues := eigenMatrix.Vectors()
	imageCount, rank := eigenValues.Dims() // eigenVectors.Cols()
	//rank := eigenVectors.Rows()
	f.Eigen = make([][]float64, (f.Height * f.Width))
	for i := 0; i < (f.Height * f.Width); i++ {
		f.Eigen[i] = make([]float64, len(f.Diff))
	}

	for i := 0; i < (rank -1); i++ {
		sumSquare := 0.0
		for j := 0; j < (f.Width * f.Height); j++ {
			for k := 0; k < imageCount; k++ {
				f.Eigen[j][i] += f.Diff[k].Pixels[j] * eigenValues.At(i, k)
			}
			sumSquare += f.Eigen[j][i] * f.Eigen[j][i]
		}
		norm := math.Sqrt(float64(sumSquare))
		for j := 0; j < len(f.Diff); j++ {
			f.Eigen[j][i] /= norm
		}
	}
}

func (f *FaceVectors) ComputeWeights(diffImagePixels FaceVector) *FaceVector {
	weight := &FaceVector{Width: f.Width, Height: f.Height}
	weight.Pixels = makeArray(len(f.Pixels))

	for nbpixels := 0; nbpixels < (f.Width * f.Height); nbpixels++ {
		for nbimages := 0; nbimages < len(f.Pixels); nbimages++ {

			weight.Pixels[nbimages] += diffImagePixels.Pixels[nbpixels] * f.Eigen[nbpixels][nbimages]
		}
	}
	return weight
}

func (f *FaceVectors) ReconstructImageWithEigenFaces(weights FaceVector) FaceVector {
	val := f.Width * f.Height
	reconstructedPixels := FaceVector{Height: f.Height, Width: f.Width}
	reconstructedPixels.Pixels = makeArray(f.Width * f.Height)
	for nbimages := 0; nbimages < len(f.Pixels); nbimages++ {
		for nbpixels := 0; nbpixels < (f.Width * f.Height); nbpixels++ {
			reconstructedPixels.Pixels[nbpixels] += weights.Pixels[nbimages] * f.Eigen[nbpixels][nbimages]
		}
	}

	for nbpixels := 0; nbpixels < val; nbpixels++ {
		reconstructedPixels.Pixels[nbpixels] += f.Mean.Pixels[nbpixels]
	}

	min := float64(math.MaxFloat64)
	max := float64(-math.MaxFloat64)

	for nbpixels := 0; nbpixels < val; nbpixels++ {
		min = math.Min(min, reconstructedPixels.Pixels[nbpixels])
		max = math.Max(max, reconstructedPixels.Pixels[nbpixels])
	}
	normalizedReconstructedPixels := FaceVector{Height: f.Height, Width: f.Width}
	normalizedReconstructedPixels.Pixels = makeArray(f.Width * f.Height)
	for nbpixels := 0; nbpixels < val; nbpixels++ {
		value := (255.0 * (reconstructedPixels.Pixels[nbpixels] - min)) / (max - min)
		normalizedReconstructedPixels.Pixels[nbpixels] = value
	}

	return normalizedReconstructedPixels
}

func (f *FaceVectors) ComputeDifferencePixels(subjectPixels FaceVector) FaceVector {
	diffPixels := FaceVector{Height: f.Height, Width: f.Width}
	diffPixels.Pixels = makeArray(f.Width * f.Height)
	for i := 0; i < (f.Width * f.Height); i++ {
		diffPixels.Pixels[i] = subjectPixels.Pixels[i] - f.Mean.Pixels[i]
	}
	return diffPixels
}

func (f *FaceVectors) ComputeDistance(subject FaceVector) (FaceVector, float64) {
	diffPixels := f.ComputeDifferencePixels(subject)
	weight := f.ComputeWeights(diffPixels)
	reconstructedEigenPixels := f.ReconstructImageWithEigenFaces(*weight)

	return diffPixels, f.ComputeImageDistance(subject, reconstructedEigenPixels)
}

func (f *FaceVectors) ComputeImageDistance(pixels1, pixels2 FaceVector) float64 {
	distance := 0.0
	for i := 0; i < (f.Width * f.Height); i++ {
		diff := pixels1.Pixels[i] - pixels2.Pixels[i]
		distance += diff * diff
	}

	return math.Sqrt(distance / float64(f.Width*f.Height))
}
func ComputeImageDistance(pixels1, pixels2 FaceVector) float64 {

	distance := 0.0
	for i := 0; i < (pixels1.Width * pixels1.Height); i++ {
		diff := pixels1.Pixels[i] - pixels2.Pixels[i]
		distance += diff * diff
	}

	return math.Sqrt(distance / float64(100))
}

func Average(faces []FaceVector) FaceVector {
	width := faces[0].Width
	height := faces[0].Height
	avg := make([]float64, width*height)

	for i := 0; i < len(faces); i++ {
		face := faces[i]
		if face.Width != width || face.Height != height {
			return FaceVector{}
		}
		for j := 0; j < width*height; j++ {
			// TODO check what this does to precision
			avg[j] += face.Pixels[j]
		}
	}

	for j := 0; j < width*height; j++ {
		avg[j] = avg[j] / float64(len(faces))
	}
	return FaceVector{
		Width:  width,
		Height: height,
		Pixels: avg,
	}
}

func Difference(face1, face2 FaceVector) FaceVector {
	width := face1.Width
	height := face1.Height
	diff := make([]float64, width*height)
	for i := 0; i < width*height; i++ {
		diff[i] += face1.Pixels[i] - face2.Pixels[i]
	}
	return FaceVector{
		Width:  width,
		Height: height,
		Pixels: diff,
	}
}

func Normalize(faces []FaceVector) []FaceVector {
	faceDiffs := make([]FaceVector, len(faces))

	avg := Average(faces)

	for i := 0; i < len(faces); i++ {
		faceDiffs[i] = Difference(faces[i], avg)
	}

	return faceDiffs
}

func transpose(mat [][]float64) (t [][]float64) {
	height := len(mat)
	width := len(mat[0])

	t = makeMat(height, width)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			t[i][j] = mat[j][i]
		}
	}
	return
}

func makeMat(width, height int) (mat [][]float64) {
	mat = make([][]float64, height)
	for i := 0; i < height; i++ {
		mat[i] = make([]float64, width)
	}
	return
}
