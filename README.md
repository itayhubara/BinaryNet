Deep Networks on classification tasks using Torch
=================================================
This is a complete training example for BinaryNets using Binary-Backpropagation algorithm as explained in
"Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio'
on following datasets: Cifar10/100, SVHN, MNIST

## Data
We use dp library to extract all the data please view installation section

## Dependencies
* Torch (http://torch.ch)
* "DataProvider.torch" (https://github.com/eladhoffer/DataProvider.torch) for DataProvider class.
* "cudnn.torch" (https://github.com/soumith/cudnn.torch) for faster training. Can be avoided by changing "cudnn" to "nn" in models.
* "dp" (https://github.com/nicholas-leonard/dp.git) for data extraction
* "unsup" (https://github.com/koraykv/unsup.git) for data pre-processing

To install all dependencies (assuming torch is installed) use:
```bash
luarocks install https://raw.githubusercontent.com/eladhoffer/DataProvider.torch/master/dataprovider-scm-1.rockspec
luarocks install cudnn
luarocks install dp
luarocks install unsup
```

## Training
Create pre-processing folder:
```lua
cd BinaryNet
mkdir PreProcData
```

Start training using:
```lua
th Main_BinaryNet_Cifar10.lua -network BinaryNet_Cifar10_Model
or,
```lua
th Main_BinaryNet_MNIST.lua -network BinaryNet_MNIST_Model
```

##Additional flags
|Flag             | Default Value        |Description
|:----------------|:--------------------:|:----------------------------------------------
|modelsFolder     |  ./Models/           | Models Folder
|network          |  Model.lua           | Model file - must return valid network.
|LR               |  0.1                 | learning rate
|LRDecay          |  0                   | learning rate decay (in # samples
|weightDecay      |  1e-4                | L2 penalty on the weights
|momentum         |  0.9                 | momentum
|batchSize        |  128                 | batch size
|stcNeurons       |  true                | using stochastic binarization for the neurons or not
|stcWeights       |  false               | using stochastic binarization for the weights or not
|optimization     |  adam                | optimization method
|SBN              |  true                | use shift based batch-normalization or not
|runningVal       |  true                | use running mean and std or not
|epoch            |  -1                  | number of epochs to train (-1 for unbounded)
|threads          |  8                   | number of threads
|type             |  cuda                | float or cuda
|devid            |  1                   | device ID (if using CUDA)
|load             |  none                |  load existing net weights
|save             |  time-identifier     | save directory
|dataset          |  Cifar10             | Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST
|dp_prepro        |  false               | preprocessing using dp lib
|whiten           |  false               | whiten data
|augment          |  false               | Augment training data
|preProcDir       |  ./PreProcData/      | Data for pre-processing (means,Pinv,P)
