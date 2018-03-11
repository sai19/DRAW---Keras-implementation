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
A = DRAW(**model_params)
A.build(input_shape=[28,28,1],read_window=2,write_window=5,rnn_type='LSTM')
A.model.summary()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_train[x_train>0.5] = 1
x_train[x_train<=0.5] = 0
print(np.max(x_train))
x_train = x_train.reshape(x_train.shape[0], 28*28)
A.model.load_weights("weights.h5")
A.build_predict()
print(A.read_dense.get_weights())
x_out = A.model_predict.predict(np.zeros((2,128)))
#print(x_out)
x_out1 = x_out[0,:]
print(np.max(x_out1))
#x_out2 = x_train[0,:]
x_out1 = x_out1*255
x_out1 = x_out1.astype('uint8')
#x_out2 = x_out2*255
#x_out2 = x_out2.astype('uint8')
#x_out2 = x_out2.reshape((28,28))
x_out1 = x_out1.reshape((28,28))
#cv2.imwrite("img_{0}.jpg".format(5+i),cv2.resize(x_out1,(280,280)))
cv2.imshow("img2",cv2.resize(x_out1,(280,280)))
cv2.waitKey(0)