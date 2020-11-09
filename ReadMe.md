## MTCNN demo for onnx model

### Step 1 convert onnx from caffe mtcnn [model](https://github.com/DuinoDu/mtcnn)
+ You need install [caffe-onnx](https://github.com/htshinichi/caffe-onnx) tools and git clone caffe mtcnn [model](https://github.com/DuinoDu/mtcnn)

```
python3 convert2onnx.py det1.prototxt det1.caffemodel det1.onnx ./
```

So,You can get 3 onnx models,`det1.onnx`,`det2.onnx` ,`det3.onnx`

### Step 2 optimizer onnx model 
```
python3 onnx_optimizer.py
```
+ Note: modify your onnx path.

### Step 3 run onnx model
```
python3 onnx_mtcnn_caffe.py --usb 0
```
