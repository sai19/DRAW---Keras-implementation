from __future__ import division
from keras.layers import LSTM,Input,Lambda,Dense,Activation
from keras import backend as K
from keras.models import Model
import keras
from keras import metrics
from keras.datasets import mnist
import numpy as np
import cv2
from layers import DRAW

model_params = {}
model_params['read_window'] = 2
model_params['write_window'] = 5
model_params['n'] = 10
latent_space_dim = 128
A = DRAW(**model_params)
A.build(input_shape=[28,28,1],read_window=2,write_window=5,rnn_type='LSTM')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_train[x_train>0.5] = 1
x_train[x_train<=0.5] = 0
x_train = x_train.reshape(x_train.shape[0], 28*28)
for i in range(20):
	A.model.fit([x_train,np.zeros((x_train.shape[0],28*28),dtype="float32"),np.zeros((x_train.shape[0],latent_space_dim),dtype="float32")],verbose=1,batch_size=200,epochs=10)
	A.model.save_weights("weights.h5")