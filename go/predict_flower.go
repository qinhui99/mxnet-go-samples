package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"sort"
	"strconv"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/songtianyi/go-mxnet-predictor/mxnet"
	"github.com/songtianyi/go-mxnet-predictor/utils"
)

var filename = flag.String("file", "japanese-anemone.jpg", "A image file to be predicted.")
var myMap map[string]string = map[string]string{}

func fillLabels() {
	file, err := os.Open("../flower102_model/labels.txt")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()
	reader := csv.NewReader(file)
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println("Error:", err)
			return
		}
		myMap[record[0]] = record[1]
	}
}

func getLabel(key int) string {

	return myMap[strconv.Itoa(key)]
}

func main() {
	flag.Parse()
	// load mean image from file
	nd, err := mxnet.CreateNDListFromFile("../flower102_model/mean.bin")
	if err != nil {
		panic(err)
	}

	// free ndarray list operator before exit
	defer nd.Free()

	// get mean image data from C memory
	item, err := nd.Get(0)
	if err != nil {
		panic(err)
	}
	fmt.Println(item.Key, item.Data[0:10], item.Shape, item.Size)

	// load model
	symbol, err := ioutil.ReadFile("../flower102_model/102flowers-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("../flower102_model/102flowers-0260.params")
	if err != nil {
		panic(err)
	}

	// create predictor
	p, err := mxnet.CreatePredictor(symbol,
		params,
		mxnet.Device{mxnet.CPU_DEVICE, 0},
		[]mxnet.InputNode{{Key: "data", Shape: []uint32{1, 3, 299, 299}}},
	)
	if err != nil {
		panic(err)
	}
	defer p.Free()

	// load test image for predction
	img, err := imgio.Open(*filename)
	if err != nil {
		panic(err)
	}

	// preprocess
	resized := transform.Resize(img, 299, 299, transform.Linear)
	res, err := utils.CvtImageTo1DArray(resized, item.Data)
	if err != nil {
		panic(err)
	}

	// set input
	if err := p.SetInput("data", res); err != nil {
		panic(err)
	}
	// do predict
	if err := p.Forward(); err != nil {
		panic(err)
	}
	// get predict result
	data, err := p.GetOutput(0)
	if err != nil {
		panic(err)
	}
	idxs := make([]int, len(data))
	for i := range data {
		idxs[i] = i
	}
	as := utils.ArgSort{Args: data, Idxs: idxs}
	sort.Sort(as)

	fmt.Println("Best result:", as.Args[0], as.Idxs[0])
	fillLabels()
	fmt.Println("label is ", getLabel(as.Idxs[0]))
}
