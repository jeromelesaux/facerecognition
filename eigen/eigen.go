package eigenface

import (
	"facerecognition/logger"
	//	"github.com/gonum/matrix/mat64"
	gomat "github.com/skelterjohn/go.matrix"
	//	"log"
	"math"
	"strconv"
)

type FaceVector struct {
	Pixels        []float64
	Width, Height int
}

type FaceVectors struct {
	Cols         int
	Rows         int
	NumberImages int
	Pixels       [][]float64
	Mean         []float64
	Weight       []float64
	Diff         [][]float64
	Cov          [][]float64
	Eigen        [][]float64
}

func makeArray(length int) []float64 {
	return make([]float64, length)
}
func makeArrays(nbArray int, length int) [][]float64 {
	a := make([][]float64, nbArray)
	for i := 0; i < nbArray; i++ {
		a[i] = makeArray(length)
	}
	return a

}

func NewFaceVectors(height, width int, nbImage int) *FaceVectors {
	rows := nbImage
	cols := height * width
	facevectors := &FaceVectors{Rows: rows, Cols: cols, NumberImages: nbImage}
	facevectors.Pixels = makeArrays(rows, cols)
	facevectors.Diff = makeArrays(rows, cols)
	facevectors.Cov = makeArrays(cols, cols)
	facevectors.Mean = make([]float64, rows, cols)
	facevectors.Eigen = makeArrays(rows, cols)
	return facevectors
}

func (f *FaceVector) ToString() string {
	return "width:" + strconv.Itoa(f.Width) + ",Height:" + strconv.Itoa(f.Height)
}

func (f *FaceVectors) Train() {
	logger.Log("Starting training")
	logger.Log("Starting mean compute")
	f.ComputeMean()
	logger.Log("Starting matrix pixel")
	f.ComputeDifferenceMatrixPixels()
	logger.Log("Starting covariance compute")
	f.ComputeCovarianceMatrix()
	logger.Log("Starting eigen faces compute")
	f.ComputeEigenFaces()
	logger.Log("Starting weights compute")
	f.ComputeWeights(f.Mean)
	logger.Log("Training ended")
}

func (f *FaceVectors) ComputeMean() {
	for i := 0; i < f.Rows; i++ {
		sum := 0.
		for j := 0; j < f.Cols; j++ {
			sum += f.Pixels[i][j]
		}
		f.Mean[i] = sum / float64(f.Cols)
	}
	return
}

func (f *FaceVectors) ComputeDifferenceMatrixPixels() {

	for i := 0; i < f.Rows; i++ {
		for j := 0; j < f.Cols; j++ {
			f.Diff[i][j] = f.Pixels[i][j] - f.Mean[i]
		}
	}
	return
}

func (f *FaceVectors) ComputeCovarianceMatrix() {
	for i := 0; i < f.Cols; i++ {
		for j := 0; j < f.Cols; j++ {
			sum := 0.0
			for k := 0; k < f.Rows; k++ {
				sum += f.Diff[k][i] * f.Diff[k][j]
			}
			f.Cov[i][j] = sum
		}
	}
	return
}

func (f *FaceVectors) ComputeEigenFaces() {
	//	mat := transpose(f.Cov)
	//denseMat := mat64.NewDense(f.Cols, f.Cols, f.Cov)
	//var eigen mat64.Eigen
	//if ok := eigen.Factorize(denseMat, true); !ok {
	//	log.Fatal("Could not factorize the eigen matrix.")
	//}
	//
	//eigenVectors := eigen.Vectors()
	//imageCount, rank := eigenVectors.Dims()
	//
	//for i := 0; i < rank; i++ {
	//	sumSquare := 0.0
	//	for j := 0; j < f.Rows; j++ {
	//		for k := 0; k < imageCount; k++ {
	//			f.Eigen[j][i] += f.Diff[j][k] * eigenVectors.At(i, k)
	//		}
	//		sumSquare += f.Eigen[j][i] * f.Eigen[j][i]
	//	}
	//	norm := math.Sqrt(float64(sumSquare))
	//	for j := 0; j < f.Rows; j++ {
	//		f.Eigen[j][i] /= norm
	//	}
	//}
	denseMat := gomat.MakeDenseMatrixStacked(f.Cov)
	eigenVectors, _, _ := denseMat.Eigen()

	imageCount := eigenVectors.Cols()
	rank := eigenVectors.Rows()
	for i := 0; i < rank; i++ {
		sumSquare := 0.0
		for j := 0; j < f.NumberImages; j++ {
			for k := 0; k < imageCount; k++ {

				f.Eigen[j][i] += f.Diff[j][k] * eigenVectors.Get(i, k)
			}
			sumSquare += f.Eigen[j][i] * f.Eigen[j][i]
		}
		norm := math.Sqrt(float64(sumSquare))
		for j := 0; j < f.NumberImages; j++ {
			f.Eigen[j][i] /= norm
		}
	}
}

func (f *FaceVectors) ComputeWeights(diffImagePixels []float64) []float64 {
	weight := make([]float64, f.Cols, f.Rows)

	for i := 0; i < f.Cols; i++ {
		for j := 0; j < f.Rows; j++ {

			weight[i] += diffImagePixels[j] * f.Eigen[i][j]
		}
	}
	return weight
}

func (f *FaceVectors) ReconstructImageWithEigenFaces(weights []float64) []float64 {
	reconstructedPixels := make([]float64, f.Rows, f.Rows)
	for i := 0; i < f.Cols; i++ {
		for j := 0; j < f.Rows; j++ {
			reconstructedPixels[j] += weights[i] * f.Eigen[j][i]
		}
	}

	for i := 0; i < f.Rows; i++ {
		reconstructedPixels[i] += f.Mean[i]
	}

	min := float64(math.MaxFloat64)
	max := float64(-math.MaxFloat64)

	for i := 0; i < f.Rows; i++ {
		min = math.Min(min, reconstructedPixels[i])
		max = math.Max(max, reconstructedPixels[i])
	}
	normalizedReconstructedPixels := make([]float64, f.Rows, f.Rows)
	for i := 0; i < f.Rows; i++ {
		normalizedReconstructedPixels[i] = (255.0 * (reconstructedPixels[i] - min)) / (max - min)
	}

	return normalizedReconstructedPixels
}

func (f *FaceVectors) ComputeDifferencePixels(subjectPixels []float64) []float64 {
	diffPixels := make([]float64, f.Rows, f.Rows)
	for i := 0; i < f.Rows; i++ {
		diffPixels[i] = subjectPixels[i] - f.Mean[i]
	}
	return diffPixels
}

func (f *FaceVectors) ComputeDistance(subject []float64) ([]float64, float64) {
	diffPixels := f.ComputeDifferencePixels(subject)
	weight := f.ComputeWeights(diffPixels)
	reconstructedEigenPixels := f.ReconstructImageWithEigenFaces(weight)

	return diffPixels, f.ComputeImageDistance(subject, reconstructedEigenPixels)
}

func (f *FaceVectors) ComputeImageDistance(pixels1, pixels2 []float64) float64 {
	distance := 0.0
	for i := 0; i < f.Rows; i++ {
		diff := pixels1[i] - pixels2[i]
		distance += diff * diff
	}
	return math.Sqrt(distance / float64(f.Rows))
}
func ComputeDifferencePixels(pixels1, pixels2 []float64) float64 {

	distance := 0.0
	for i := 0; i < (len(pixels1) * len(pixels1)); i++ {
		diff := pixels1[i] - pixels2[i]
		distance += diff * diff
	}

	return math.Sqrt(distance / float64(100))
}

func Average(faces [][]float64, width int, height int) []float64 {
	avg := make([]float64, width*height)

	for i := 0; i < len(faces); i++ {
		face := faces[i]

		for j := 0; j < width*height; j++ {
			// TODO check what this does to precision
			avg[j] += face[j]
		}
	}

	for j := 0; j < width*height; j++ {
		avg[j] = avg[j] / float64(len(faces))
	}
	return avg
}

func Difference(face1, face2 []float64, width int, height int) []float64 {

	diff := make([]float64, width*height)
	for i := 0; i < width*height; i++ {
		diff[i] += face1[i] - face2[i]
	}
	return diff
}

func Normalize(faces [][]float64, width int, height int) [][]float64 {
	faceDiffs := make([][]float64, len(faces))

	avg := Average(faces, width, height)

	for i := 0; i < len(faces); i++ {
		faceDiffs[i] = Difference(faces[i], avg, width, height)
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

func transposeMatrix(mat []float64, width, height int) (t []float64) {
	t = make([]float64, width*height)
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			t[i+j] = mat[j+i]
		}
	}
	return
}

//func Eigenfaces(faces [][]float64, width int, height int) []float64 {
//	mat := makeMat(width, height)
//	mat = transpose(faces)
//	epsilon := 0.01
//	small := 0.01
//
//	eigenvalues := mat64.SVD{mat, epsilon, small, true /*wantu*/, false /*wantv*/}.Values()
//	return eigenvalues
//}

func makeMat(width, height int) (mat [][]float64) {
	mat = make([][]float64, height)
	for i := 0; i < height; i++ {
		mat[i] = make([]float64, width)
	}
	return
}
