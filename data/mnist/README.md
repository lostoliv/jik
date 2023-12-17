# MNIST Dataset Tools

## MNIST Render

### Info

This script generates N random single digit images.
The digit in each image is randomly sized and placed while some distortion and
noise gets applied.

See <a href="http://yann.lecun.com/exdb/mnist" target="_blank">here</a> for
the original dataset.

### Requirements

This python script is mostly using numpy and opencv modules.
Make sure you have both of them.

### Usage

To get help, just type:
```sh
python3 mnist_render.py -h
```

Try for example:
```sh
python3 mnist_render.py -out . -num 1024 -seed 100 -dmax 1.0 -gnmax 0.4 -spnmax 0.1 -concat
```

It will generate an image:
```sh
mnist.png
```
with 1024 random digit images with some distortion, gaussian and salt-pepper
noise.

Note that the default image resolution is 28x28.
1024 images (32x32 images matrix) concatenated into one image will have a
resolution of 896x896.

To generate a fully compatible MNIST dataset (60k training images and 10k
testing images) you can try:
```sh
python3 mnist_render.py -out . -num 60000 -seed 101 -dmax 1.0 -dataset -prefix train
python3 mnist_render.py -out . -num 10000 -seed 102 -dmax 1.0 -dataset -prefix t10k
```

Note that we removed the noise here to match the original MNIST dataset.
We left however some distortion.

It will generate 4 files (2 training files, images and labels, and 2 testing
files):
```sh
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

Make sure the seed is different for the training and testing datasets so all
images are unique (i.e. no training image ends up in the testing set as well).

You can start training this dataset with
<a href="https://www.tensorflow.org" target="_blank">Tensorflow</a> or
<a href="http://caffe.berkeleyvision.org" target="_blank">Caffe</a>.

You can also have these files gzipped (like the original dataset) by adding:
```sh
-gz
```
as a flag to the command line (a .gz extension will be added automatically to
the file name).

## SVHN To MNIST

### Info

This script converts the SVHN (Street View House Numbers) dataset to a MNIST
dataset format.
Note that, by default, the final resolution will be 28x28 (down from 32x32)
and grayscale (down from RGB).

See <a href="http://ufldl.stanford.edu/housenumbers" target="_blank">here</a>
for the original dataset.

### Requirements

This python script is mostly using numpy, scipy and opencv.
Make sure you have all of them.

### Usage

To get help, just type:
```sh
python3 svhn_to_mnist.py -h
```

Try for example the following to convert the SVHN training set:
```sh
curl http://ufldl.stanford.edu/housenumbers/test_32x32.mat
python3 svhn_to_mnist.py -svhn test_32x32.mat -out . -prefix t10k
```

## SVHN To CIFAR-10

### Info

This script converts the SVHN (Street View House Numbers) dataset to a
CIFAR-10 dataset format.

See <a href="http://ufldl.stanford.edu/housenumbers" target="_blank">here</a>
for the original dataset.

### Requirements

This python script is mostly using numpy, scipy and opencv.
Make sure you have all of them.

### Usage

To get help, just type:
```sh
python3 svhn_to_cifar10.py -h
```

Try for example the following to convert the SVHN training set:
```sh
curl -O http://ufldl.stanford.edu/housenumbers/test_32x32.mat
python3 svhn_to_cifar10.py -svhn test_32x32.mat -out . -prefix test
```
