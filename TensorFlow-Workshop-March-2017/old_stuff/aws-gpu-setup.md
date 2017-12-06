Get setup and configured with an AWS GPU backed instance.
These instructions show how to train a 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) classifier and are
based on other people's writeups - they can be found 
[here](http://markus.com/install-theano-on-aws/) and 
[here](http://ramhiser.com/2016/01/05/installing-tensorflow-on-an-aws-ec2-instance-with-gpu-support/).

### 1. AWS SETUP

Setup an AWS ubuntu instance (with gpu). Log in.

```bash
# username for ubuntu instance is 'ubuntu'
ssh -i [path/to/key.pem] ubuntu@[DNS]
```

### 2. THEANO

Install first batch of system deps.

```bash
sudo apt-get install -y \
    gcc \
    g++ \
    gfortran 
    build-essential \
    git \
    wget \
    linux-generic \
    libopenblas-dev \
    python-dev \
    python-pip \
    python-nose \
    python-numpy \
    python-scipy
```

install 'bleeding edge theano'.

```bash
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
```

### 3. CUDA

get the cuda debian package deb file. 

```bash
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb  
```

register it via dpkg. 

```bash
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
```

update the system so the cuda package is visible to apt-get.

```bash
sudo apt-get update
```

install cuda.

```bash
sudo apt-get install -y cuda
```

export cuda bin to path. export include path for header files.
```base
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc
```

reboot.

```bash
sudo reboot
```

the cuda bin directory that is now on the path contains this script.
run it in home (~).

```bash
cuda-install-samples-7.5.sh ~/
```

it will create a 'Samples' directory. cd to the following.

```bash
cd NVIDIA_CUDA-7.5_Samples/1_Utilities/deviceQuery  
```

make and run.

```bash
make  
./deviceQuery
```

create a .theanorc set with gpu flags.

```bash
echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc
```

### 4. KERAS

this setup uses keras. 
keras uses HDF5 to save files. 
download the system deps.

```bash
# http://stackoverflow.com/questions/24744969/installing-h5py-on-an-ubuntu-server
sudo apt-get install libhdf5-dev
```

install keras and h5py. h5py is a python client for HDF5.

```bash
sudo pip install keras
sudo pip install h5py
```

### 5. CUDNN

cudnn reduces the train time of conv nets.
sign up for [nvidia dev program](https://developer.nvidia.com/rdp/cudnn-download).
unzip the cudnn download.

```bash
tar -zxf cudnn-7.0.tgz && rm cudnn-7.0.tgz
```

the unzipped directory should have a lib and an include.
copy those in to respective directories of the cuda installation.

```bash
sudo cp -R cudnn-7.0/lib64/* /usr/local/cuda/lib64/
sudo cp cudnn-7.0/include/cudnn.h /usr/local/cuda/include/
```

reboot.

```bash
sudo reboot
```

### 6. OTHER

#### how to keep procs running after ending ssh session

[screen - ask ubuntu](http://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session)

[screen - digital ocean](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-screen-on-an-ubuntu-cloud-server)

### 7. INTEGRATION

run the the py files in this repo to sanity check various aspects of the previous
six steps.


### 8. TRAIN

```python
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

batch_size = 32
nb_classes = 10
nb_epoch = 200

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('Start')
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)

print('Saving')
json_string = model.to_json()
open('model_arch.json', 'w').write(json_string)
model.save_weights('model_weights.h5')

```
