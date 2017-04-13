package main

import (
	"fmt"
	"io/ioutil"
	"sort"

	"github.com/songtianyi/go-mxnet-predictor/mxnet"
	"github.com/songtianyi/go-mxnet-predictor/utils"
)

func main() {
	// load model
	symbol, err := ioutil.ReadFile("../iris_model/iris-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("../iris_model/iris-0015.params")
	if err != nil {
		panic(err)
	}

	// create predictor
	p, err := mxnet.CreatePredictor(symbol,
		params,
		mxnet.Device{mxnet.CPU_DEVICE, 0},
		[]mxnet.InputNode{
			mxnet.InputNode{Key: "data", Shape: []uint32{1, 4}},
			mxnet.InputNode{Key: "softmax_label", Shape: []uint32{1, 4}},
		},
	)
	defer p.Free()
	if err != nil {
		panic(err)
	}
	fmt.Println("OK! We get the preditor.")
	//	inputData := []float32{5.5, 2.3, 4.0, 1.3} //label=1,predict=2 [1.6753012e-19 5.440593e-18 1] [0 1 2]
	inputData := []float32{5.0, 3.3, 1.4, 0.2} //label=0,predict=1  [0.9980082 0.001988203 3.529224e-06] [1 0 2]
	//inputData := []float32{5.8, 2.7, 5.1, 1.9} //label=2,predict=2
	//	inputData := [...][2]float32{{5.1, 3.8, 1.9, 0.4}, {5.5, 2.3, 4.0, 1.3}}
	// set input
	if err := p.SetInput("data", inputData); err != nil {
		panic(err)
	}
	// do predict
	if err := p.Forward(); err != nil {
		panic(err)
	}
	fmt.Println("OK! We are prediting...")
	//	 get predict result
	data, err := p.GetOutput(0)
	if err != nil {
		panic(err)
	}
	idxs := make([]int, len(data))
	for i := range data {
		idxs[i] = i
	}
	fmt.Println(data, idxs)
	as := utils.ArgSort{Args: data, Idxs: idxs}
	sort.Sort(as)
	fmt.Println("result:")
	fmt.Println("posibility:", as.Args)
	fmt.Println("label:", as.Idxs)
	fmt.Println("Best result:", as.Args[0], as.Idxs[0])

}
