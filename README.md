# jik

JIK is a simple and lightweight deep learning library.

It was written between March and April 2016.
It was solely developed for me to understand the inner workings of neural
networks and backward propagation.

It is very limited to keep everything simple. Tho you can still train a good
MNIST/CIFAR10 classifier with it with a good accuracy in a few seconds/minutes.

To get going quickly, run this:
```sh
zsh build.sh ; zsh data/download/mnist.sh ; build/sandbox/mnist/mnist -dataset data/dataset/mnist -train -fc -numstep 5000 -saveeach -1 -testeach 5000
```

It will compile the code, download the MNIST dataset and train a good model in
a few seconds. It should print the accuracy of the trained model at the end.

## Info

This project is implementing basic algorithms for two of the main deep
learning architectures:
* *Feed-forward Neural Networks* (FFNN) (including *Convolutional Neural
  Network* (CNN) models)
* *Recurrent Neural Networks* (RNN) (including *Long Short-Term Memory* (LSTM)
  models)

It is currently only implemented on the CPU (single-threaded) but a
multi-threaded version as well as a version with CUDA kernels would be welcome
additions.

I tried to keep the design of the system very simple and lightweight so it's
easy to parse and understand.
There's no dependency by default, making it easy to compile, port and run.

The main goal of this project is to understand the nuts and bolts of the math
behind deep learning, particularly around automatic differentiation, backward
propagation and optimization.
I did not focus on efficiency but rather on readability.

For better efficiency, one would use linear algebra libraries for low level
math functions with GPU kernels, especially for convolutions and other
vector/matrix or matrix/matrix operations. For example one could use eigen or
lapack libraries like BLAS or GPU-based solutions like cuBLAS or low-level
deep neural network primitives from cuDNN.

This is beyond the scope of this project but would be great to have.

Running the same CNN model to classify MNIST in JIK and TensorFlow using a
batch size of 128 over 1000 iterations with a RMSprop optimizer, we get using
a high end 2015 Macbook Pro (2.8 Ghz i7 Intel processor):

JIK             : time for 1000 steps: 2min 13sec, test accuracy: 97.95%

TensorFlow (CPU): time for 1000 steps: 0min 30sec, test accuracy: 98.32%

Using a fully-connected architecture (instead of CNN), using the same
hardware and hyper-parameters, we get:

JIK             : time for 1000 steps: 0min 14sec, test accuracy: 95.85%

TensorFlow (CPU): time for 1000 steps: 0min 04sec, test accuracy: 89.21%

All these results were averaged over 10 runs.

## Structure

* core: main library, including layers, graph, model and solver
* recurrent: RNN, including LSTM
* data: placeholder for the datasets (with some scripts to download them)
* model: pre-trained models
* sandbox: examples
  * linear_regression: scale model trying to learn linear regression
  * mnist            : mnist classifier (classifying the mnist dataset)
  * cifar10          : cifar10 classifier (classifying the cifar10 dataset)
  * textgen          : RNN (or LSTM) model taking an input text and predicting
                       sentences

## Requirements

You must have a Linux, macOS or Windows system with:
* some C++ compiler compatible with C++ 11 (see below)
* cmake (version 2.8 or above)

## Compiler

By default, we will use gcc/g++ to compile anything.
To use clang/clang++, just define those environment variables:
* export CC=clang    (default = gcc)
* export CXX=clang++ (default = g++)

Please note that on macOS, gcc/g++ are just symlinks to clang/clang++.

You need a C++ 11 (aka C++ 0x) friendly C++ compiler.

## Compilation

This project is using cmake.
Make sure you have at least version 2.8.

Below we use `zsh` but please use your favorite shell script (everything
should work with `bash`).

From the root directory, run:
```sh
zsh build.sh
```

Which is equivalent to:
```sh
mkdir build
cd build
cmake ..
make -j8
```

To build a debug version, run:
```sh
zsh build.sh debug
```

Which is equivalent to:
```sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j8
```

To clean everything, run:
```sh
zsh build.sh clean
```

## Code style (cpplint)

We're using Google C++ style guide:
https://google.github.io/styleguide/cppguide.html

To make sure the code is compliant with this style guide, via cpplint, run:
```sh
make lint
```

## Data

From the root directory, you can run:
```sh
zsh data.sh
```
in order to get all the data needed to run the sandbox examples.

The following will be downloaded/generated:
* MNIST dataset
* CIFAR10 dataset
* A text file with Shakespeare input

To clean everything, run:
```sh
zsh data.sh clean
```

## Sandbox examples

Make sure you download the data before running the sandbox examples.

Then, from the build directory, you can run the following sandbox examples.

### Linear regression

This example will try to learn a scalar value using linear regression.
We generate bunch of input values X and output values Y so that:
  Y = N * X + eps
(eps is a small noise added to the output values to make the learning process
more difficult)
We try to learn the value of N, given the input and output values.

Training the scale model to learn value 3.14159265359:
```sh
sandbox/linear_regression/linear_regression -train -scale 3.14159265359
```

### MNIST classifier

This example will classify the MNIST dataset (see here:
http://yann.lecun.com/exdb/mnist).

Training a CNN model, without batch normalization:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist -train -name mnist_conv
```

Training a CNN model, without batch normalization, using a SGD solver (instead
of a RMSprop solver by default):
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist -train -solver sgd -name mnist_sgd_conv
```

Training a CNN model, with batch normalization:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist -train -bn -name mnist_conv_bn
```

Training a FC model:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist -train -fc -name mnist_fc
```

Testing a pre-trained CNN model:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist -model model/mnist_conv.model
```

Testing a pre-trained FC model:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist -fc -model model/mnist_fc.model
```

Fine-tuning a pre-trained CNN model:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist -model model/mnist_conv.model -train -name mnist_conv_finetune
```

Testing a pre-trained CNN model on the synthetic (rendered) MNIST dataset:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist_render -model model/mnist_conv.model
```

You can see that the model is behaving poorly here. It has been trained on
real MNIST dataset but hasn't seen any rendered data (domain gap).
Let's try to train a model on both the real and synthetic MNIST datasets
(mixed model):
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist:../data/dataset/mnist_render -train -name mnist_mix_conv
```

Let's now test this new mixed model (pre-trained) on the real MNIST dataset,
the synthetic MNIST dataset and both at the same time:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist -model model/mnist_mix_conv.model
sandbox/mnist/mnist -dataset ../data/dataset/mnist_render -model model/mnist_mix_conv.model
sandbox/mnist/mnist -dataset ../data/dataset/mnist:../data/dataset/mnist_render -model model/mnist_mix_conv.model
```

You can see that it's now giving very accurate results on all datasets.
We did the same thing for the FC model (mixed model):
```sh
sandbox/mnist/mnist -dataset ../data/dataset/mnist -fc -model model/mnist_mix_fc.model
sandbox/mnist/mnist -dataset ../data/dataset/mnist_render -fc -model model/mnist_mix_fc.model
sandbox/mnist/mnist -dataset ../data/dataset/mnist:../data/dataset/mnist_render -fc -model model/mnist_mix_fc.model
```

### SVHN classifier (MNIST-based)

This example will classify the SVHN dataset (see here:
http://ufldl.stanford.edu/housenumbers).

Since we converted the dataset to a MNIST dataset format, we will use the
MNIST classifier here as well.

Training a CNN model, without batch normalization:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/svhn -train -name svhn_mnist_conv
```

Training a CNN model, without batch normalization, using a SGD solver (instead
of a RMSprop solver by default):
```sh
sandbox/mnist/mnist -dataset ../data/dataset/svhn -train -solver sgd -name svhn_mnist_sgd_conv
```

Training a CNN model, with batch normalization:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/svhn -train -bn -name svhn_mnist_conv_bn
```

Training a FC model:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/svhn -train -fc -name svhn_mnist_fc
```

Testing a pre-trained CNN model:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/svhn -model model/svhn_mnist_conv.model
```

Testing a pre-trained FC model:
```sh
sandbox/mnist/mnist -dataset ../data/dataset/svhn -fc -model model/svhn_mnist_fc.model
```

### SVHN classifier (CIFAR-10-based)

This example will classify the SVHN dataset (see here:
http://ufldl.stanford.edu/housenumbers).

Since we converted the dataset to a CIFAR-10 dataset format, we will use the
CIFAR-10 classifier here as well.

Training a model, without batch normalization:
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/svhn -train -name svhn_cifar10
```

Training a model, without batch normalization, using a SGD solver (instead of
a RMSprop solver by default):
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/svhn -train -solver sgd -name svhn_cifar10_sgd
```

Training a model, with batch normalization:
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/svhn -train -bn -name svhn_cifar10_bn
```

Testing a pre-trained model:
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/svhn -model model/svhn_cifar10.model
```

### CIFAR10 classifier

This example will classify the CIFAR10 dataset (see here:
http://www.cs.toronto.edu/~kriz/cifar.html).

Training a model, without batch normalization:
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/cifar10 -train -name cifar10
```

Training a model, without batch normalization, using a SGD solver (instead of
a RMSprop solver by default):
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/cifar10 -train -solver sgd -name cifar10_sgd
```

Training a model, with batch normalization:
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/cifar10 -train -bn -name cifar10_bn
```

Training a model, grayscaling the input images (from RGB):
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/cifar10 -train -gray -name cifar10_gray
```

Testing a pre-trained model:
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/cifar10 -model model/cifar10.model
```

Fine-tuning a pre-trained model:
```sh
sandbox/cifar10/cifar10 -dataset ../data/dataset/cifar10 -model model/cifar10.model -train -name cifar10_finetune
```

### Text generator

This example will take an input text file and start generating sentences with
the same style using a RNN or LSTM.
Feel free to use your own text file.

Training a RNN model:
```sh
sandbox/textgen/textgen -dataset ../data/dataset/textgen/input.txt -model rnn
```

Training a LSTM model:
```sh
sandbox/textgen/textgen -dataset ../data/dataset/textgen/input.txt -model lstm
```
