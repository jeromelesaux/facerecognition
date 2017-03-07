package model

import (
	"facerecognition/algorithm"
	"facerecognition/logger"
)

func AssignLabel(trainingSet []*ProjectedTrainingMatrix, testFace *algorithm.Matrix, k int, computeDistance func(a, b *algorithm.Matrix) float64) (string, float64) {
	neighbors := FindKNN(trainingSet, testFace, k, computeDistance)
	return Classify(neighbors)
}

func FindKNN(trainingSet []*ProjectedTrainingMatrix, testFace *algorithm.Matrix, k int, computeDistance func(a, b *algorithm.Matrix) float64) []*ProjectedTrainingMatrix {
	numOfTrainingSet := len(trainingSet)
	if k > numOfTrainingSet {
		logger.Log("K is larger than the length of trainingSet!")
		return nil
	}
	// initialization
	neighbors := make([]*ProjectedTrainingMatrix, k)
	for i := 0; i < k; i++ {
		trainingSet[i].Distance = computeDistance(trainingSet[i].Matrix, testFace)
		neighbors[i] = trainingSet[i]
	}
	// go through the remaining records in the trainingSet to find K nearest
	// neighbors
	for i := k; i < numOfTrainingSet; i++ {
		trainingSet[i].Distance = computeDistance(trainingSet[i].Matrix, testFace)
		maxIndex := 0
		for j := 0; j < k; j++ {
			if neighbors[j].Distance > neighbors[maxIndex].Distance {
				maxIndex = j
			}
		}

		if neighbors[maxIndex].Distance > trainingSet[i].Distance {
			neighbors[maxIndex] = trainingSet[i]
		}
	}
	return neighbors
}

func Classify(neighbors []*ProjectedTrainingMatrix) (string, float64) {
	mmap := make(map[string]float64)
	num := len(neighbors)
	for index := 0; index < num; index++ {
		temp := neighbors[index]
		key := temp.Label
		if val, ok := mmap[key]; ok {
			val += 1 / temp.Distance
			mmap[key] = val
		} else {
			mmap[key] = 1 / temp.Distance
		}
	}
	maxSimilarity := 0.
	returnedLabel := ""
	for label, value := range mmap {
		if value > maxSimilarity {
			maxSimilarity = value
			returnedLabel = label
		}
	}
	return returnedLabel, maxSimilarity
}
