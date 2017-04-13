

The samples are based on Mxnet and go-mxnet-predictor.  Therefore we need:

1. Mxnet .  https://github.com/dmlc/mxnet

2. go-mxnet-predictor. https://github.com/songtianyi/go-mxnet-predictor


We have three samples.

1. OXFORD 102 flower Model.  
We can use it to classify 102 flowers .

run the code:
go run predict_flower.go -file  camellia.jpg

2. Inception Model.
The model comes from http://data.mxnet.io/mxnet/models/imagenet/inception-bn/
We can use it to classify 1000 objects.

run the code:
go run predict_inception.go -file paper.jpg

3. Iris dataset Model.
Show how to classify iris dataset.

run the code:
go run predict_iris.go
