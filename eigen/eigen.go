package eigenface

//import "github.com/gonum/matrix/mat64"
import (
	"math"
	"strconv"
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
	Diff   []FaceVector
	Cov    []FaceVector
	//Eigen  []FaceVector
}

func makeArray(length int) []float64 {
	return make([]float64, length)
}

func NewFaceVectors(height, width int, nbImage int) *FaceVectors {
	facevectors := &FaceVectors{Height: height, Width: width}
	facevectors.Pixels = make([]FaceVector, nbImage)
	facevectors.Diff = make([]FaceVector, nbImage)
	facevectors.Cov = make([]FaceVector, height*width)
	//facevectors.Eigen = make([]FaceVector, nbImage)
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

	//f.ComputeEigenFaces()
}

func (f *FaceVectors) ComputeMean() {
	for k := 0; k < len(f.Pixels); k++ {
		sum := 0.
		for l := 0; l < (f.Width * f.Height); l++ {
			sum += f.Pixels[k].Pixels[l]
		}
		f.Mean.Pixels[k] = sum / float64(f.Width*f.Height)
	}
	return
}

func (f *FaceVectors) ComputeDifferenceMatrixPixels() {

	for i := 0; i < len(f.Pixels); i++ {
		f.Diff[i].Pixels = makeArray(f.Width * f.Height)
		for j := 0; j < (f.Width * f.Height); j++ {
			f.Diff[i].Pixels[j] = f.Pixels[i].Pixels[j] - f.Mean.Pixels[j]
		}
	}
	return
}

func (f *FaceVectors) ComputeCovarianceMatrix() {
	for i := 0; i < (f.Height * f.Width); i++ {
		f.Cov[i].Pixels = makeArray(f.Width * f.Height)
		for j := 0; j < (f.Height * f.Width); j++ {
			sum := 0.0
			for k := 0; k < len(f.Diff); k++ {
				sum += f.Diff[k].Pixels[i] * f.Diff[k].Pixels[j]
			}
			f.Cov[i].Pixels[j] = sum
		}
	}
	return
}

//
//func (f *FaceVectors) ComputeEigenFaces() {
//	cov := make([][]float64, len(f.Cov))
//	for i := 0; i < len(f.Cov); i++ {
//		cov[i] = make([]float64, len(f.Cov[i].Pixels))
//		for j := 0; j < len(f.Cov[i].Pixels); j++ {
//			cov[i][j] = f.Cov[i].Pixels[j]
//		}
//	}
//
//	//denseMatrix := gomat.MakeDenseMatrix(cov, len(f.Cov), len(f.Cov[0].Pixels))
//	//eigenVectors, _, _, _ := denseMatrix.SVD()
//	denseMat := gomat.MakeDenseMatrixStacked(cov)
//	//fmt.Println(denseMat)
//	eigenVectors, _, _ := denseMat.Eigen()
//	//fmt.Println(eigenVectors)
//	imageCount := eigenVectors.Cols()
//	rank := eigenVectors.Rows()
//
//	for i := 0; i < rank; i++ {
//		sumSquare := 0.0
//		for j := 0; j < (f.Width * f.Height); j++ {
//			for k := 0; k < imageCount; k++ {
//
//				f.Eigen[j].Pixels[i] += f.Diff[j].Pixels[k] * eigenVectors.Get(i, k)
//			}
//			sumSquare += f.Eigen[j].Pixels[i] * f.Eigen[j].Pixels[i]
//		}
//		norm := math.Sqrt(float64(sumSquare))
//		for j := 0; j < len(f.Diff); j++ {
//			f.Eigen[j].Pixels[i] /= norm
//		}
//	}
//}

func (f *FaceVectors) ComputeWeights(diffImagePixels FaceVector) FaceVector {
	eigenWeights := FaceVector{Height: f.Height, Width: f.Width}
	eigenWeights.Pixels = makeArray(f.Width * f.Height)
	for i := 0; i < len(f.Pixels); i++ {
		for j := 0; j < (f.Width * f.Height); j++ {
			eigenWeights.Pixels[i] += diffImagePixels.Pixels[j] * f.Pixels[i].Pixels[j]
		}
	}

	return eigenWeights

}

func (f *FaceVectors) ReconstructImageWithEigenFaces(weights FaceVector) FaceVector {
	reconstructedPixels := FaceVector{Height: f.Height, Width: f.Width}
	reconstructedPixels.Pixels = makeArray(f.Width * f.Height)
	for i := 0; i < len(f.Pixels); i++ {
		for j := 0; j < (f.Width * f.Height); j++ {
			reconstructedPixels.Pixels[j] += weights.Pixels[j] * f.Pixels[i].Pixels[j]
		}
	}

	for i := 0; i < len(f.Mean.Pixels); i++ {
		reconstructedPixels.Pixels[i] += f.Mean.Pixels[i]
	}

	min := float64(math.MaxFloat64)
	max := float64(-math.MaxFloat64)

	for i := 0; i < len(f.Mean.Pixels); i++ {
		min = math.Min(min, reconstructedPixels.Pixels[i])
		max = math.Max(max, reconstructedPixels.Pixels[i])
	}

	normalizedReconstructedPixels := FaceVector{Height: f.Height, Width: f.Width}
	normalizedReconstructedPixels.Pixels = makeArray(f.Width * f.Height)
	for i := 0; i < len(reconstructedPixels.Pixels); i++ {
		normalizedReconstructedPixels.Pixels[i] = (255.0 * (reconstructedPixels.Pixels[i] - min)) / (max - min)
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

func (f *FaceVectors) ComputeDistance(subject FaceVector) float64 {
	diffPixels := f.ComputeDifferencePixels(subject)
	weights := f.ComputeWeights(diffPixels)
	reconstructedEigenPixels := f.ReconstructImageWithEigenFaces(weights)
	return f.ComputeImageDistance(subject, reconstructedEigenPixels)
}

func (f *FaceVectors) ComputeImageDistance(pixels1, pixels2 FaceVector) float64 {

	distance := 0.0
	for i := 0; i < (f.Width * f.Height); i++ {
		diff := pixels1.Pixels[i] - pixels2.Pixels[i]
		distance += diff * diff
	}

	return math.Sqrt(distance / float64(100))
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

//
//func Eigenfaces(faces []FaceVector) [][]float64 {
//	mat := matrix(faces)
//	epsilon := 0.01
//	small := 0.01
//	eigenvalues := mat64.SVD{mat, epsilon, small, true /*wantu*/, false /*wantv*/}.Values()
//	return eigenvalues
//}

func matrix(faces []FaceVector) [][]float64 {
	mat := make([][]float64, len(faces))
	height := len(mat)
	for i := 0; i < height; i++ {
		mat[i] = faces[i].Pixels
	}
	return transpose(mat)
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
