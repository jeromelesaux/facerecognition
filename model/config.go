package model

import (
	"encoding/json"
	"github.com/jeromelesaux/facerecognition/logger"
	"os"
	"path/filepath"
	"sync"
)

type Config struct {
	FaceDetectionConfigurationFile string `json:"opencvfile"`
	FaceRecognitionBasePath        string `json:"facerecognitionbasepath`
}

func (conf *Config) GetDataLib() string {
	return conf.FaceRecognitionBasePath + separator + "data_library.json"
}

func (conf *Config) GetTmpDirectory() string {
	return conf.FaceRecognitionBasePath + separator + "tmp" + separator
}

func (conf *Config) GetFaceRecognitionBasePath() string {
	return conf.FaceRecognitionBasePath + separator
}

var (
	c              *Config
	configLoadOnce sync.Once
	configFile     string
	separator      = string(filepath.Separator)
)

func SetConfigFile(filepath string) {
	configFile = filepath
}

func SetAndLoad(filepath string) *Config {
	SetConfigFile(filepath)
	return GetConfig()
}

func GetConfig() *Config {
	configLoadOnce.Do(func() {
		load(configFile)
	})
	return c
}

func load(filepath string) *Config {
	f, err := os.Open(filepath)
	if err != nil {
		logger.Logf("cannot not open file %s error %v", filepath, err)
		return nil
	}
	defer f.Close()

	if err := json.NewDecoder(f).Decode(&c); err != nil {
		logger.Logf("error while decoding json file with error %v", err)
		return nil
	}
	logger.Logf("configuration file %s loaded with structure %v", filepath, *c)
	return c
}
