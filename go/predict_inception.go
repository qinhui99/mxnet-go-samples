package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"sort"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/songtianyi/go-mxnet-predictor/mxnet"
	"github.com/songtianyi/go-mxnet-predictor/utils"
)

var filename = flag.String("file", "cock.jpg", "A image file to be predicted.")

func main() {
	flag.Parse()

	// load mean image from file
	nd, err := mxnet.CreateNDListFromFile("../inception_model/mean_224.nd")
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
	symbol, err := ioutil.ReadFile("../inception_model/Inception-BN-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("../inception_model/Inception-BN-0126.params")

	if err != nil {
		panic(err)
	}

	// create predictor
	p, err := mxnet.CreatePredictor(symbol,
		params,
		mxnet.Device{mxnet.CPU_DEVICE, 0},

		[]mxnet.InputNode{{Key: "data", Shape: []uint32{1, 3, 224, 224}}},
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
	resized := transform.Resize(img, 224, 224, transform.Linear)
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
}
