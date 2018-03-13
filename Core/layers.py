from __future__ import division
import os
from keras.layers import LSTM,Input,Lambda,Dense,Activation
from keras import backend as K
from keras.models import Model
import keras
from keras import metrics
import numpy as np
epsilon_std = 1.0

class DRAW(object):
    ''' Draw layer '''
    def __init__(
        self,
        read_window,
        write_window,
        n,
        rnn_type="LSTM",
        h=128,
        attention = True,
        input_shape=None,
        **kwargs):
        '''
        Args:
        	attention(bool)         : whether to pay attention or not(refer to the paper)
        	n(int)                  : number of iterations to run (refer to the paper)
            read_window (int)       : read window size (refer to the paper, only valid if attention=True)
            write_window (int)		: write window size (refer to the paper, only valid if attention=True)
            rnn_type   				: type of rnn to use (supported "RNN","LSTM","GRU", default = "LSTM")
            h (int)       			: latent space dimension (refer to the paper, default = 128)
            crop_right (bool)       : if True, crop rightmost of the feature maps (mask A, introduced in [https://arxiv.org/abs/1601.06759] )
        	
        '''
        self.read_window = read_window
        self.write_window = write_window
        self.rnn_type = rnn_type
        self.h = h
        self.n = n
        self.attention = attention

    @staticmethod 
    def Latent_Distribution_Sampling(args,h):
    	# this function outputs the latent space sampling using parametrization trick
    	z_mean,z_log_var = args
    	epsilon = K.random_normal(shape=(K.shape(z_mean)[0], h), mean=0.,stddev=epsilon_std)
    	return z_mean + K.exp(z_log_var / 2) * epsilon

    @staticmethod
    def attn_window(args,N,dim):
    	# this function returns the filter banks
    	gx,sigma2,delta = args
    	mu_x = Lambda(lambda x: (K.reshape(K.cast(K.arange(start=0,stop=N), "float32")- K.constant(N / 2 - 0.5),(N,1)))*x[0]+x[1])([delta,gx])
    	mu_x = Lambda(lambda x:K.permute_dimensions(x,(1,0)))(mu_x)
    	mu_x = Lambda(lambda x:K.expand_dims(x,axis=2))(mu_x)
    	mu_x = Lambda(lambda x:K.repeat_elements(x,axis=2,rep=dim))(mu_x)
    	a = Lambda(lambda x:K.repeat_elements(K.reshape(K.cast(K.arange(start=0,stop=dim), "float32"), (1,dim) ),axis=0,rep=N)-x)(mu_x)
    	sigma2 = keras.layers.Reshape([1, 1])(sigma2)
    	a_out = Lambda(lambda x:K.square(x))(a)
    	sigma2 = Lambda(lambda x:1/(2*(x+K.epsilon())))(sigma2)
    	a_out = keras.layers.multiply([a_out,sigma2])
    	Fx = Lambda(lambda x:K.exp(-x))(a_out)
    	Fx = Lambda(lambda x:x/(K.sum(x,2,keepdims=True)+K.epsilon()))(Fx)
    	return Fx
    @staticmethod
    def read(args,N,attention,A,B):
    	image_t,image_err,Fx,Fyt,gamma = args
    	if attention:
    		Fxt = K.permute_dimensions(Fx,(0,2,1))
    		image_t = K.reshape(image_t,(-1,B,A))
    		image_err = K.reshape(image_err,(-1,B,A))
    		image_t_dot = K.batch_dot(image_t,Fxt)
    		glimpse = K.batch_dot(Fyt,K.batch_dot(image_t,Fxt))
    		glimpse = K.reshape(glimpse,(-1,N*N))*K.reshape(gamma,(-1,1))
    		glimpse_error = K.batch_dot(Fyt,K.batch_dot(image_err,Fxt))
    		glimpse_error = K.reshape(glimpse_error,(-1,N*N))*K.reshape(gamma,(-1,1))
    		return K.concatenate([glimpse,glimpse_error],1)
    	else:
    		return K.concatenate([image_t,image_err],1)

    @staticmethod
    def concatenate(args):
    	return K.concatenate(args,1)

    @staticmethod
    def write(args,write_n,attention,A,B,write_dense_image):
    	Fxt,Fyt,gamma,h_dec_prev_out = args
    	if attention:
    		Fyt = K.permute_dimensions(Fyt,(0,2,1))
    		w = write_dense_image(h_dec_prev_out)
    		w = K.reshape(w,[-1,write_n,write_n])
    		wr = K.batch_dot(Fyt,K.batch_dot(w,Fxt))
    		wr = K.reshape(wr,[-1,B*A])
    		return wr*K.reshape(1.0/(gamma+K.epsilon()),[-1,1])
    	else:
    		return write_dense(h_dec_prev)

    
    def build(self,input_shape, read_window, write_window,rnn_type):
    	h = self.h
    	self.img_width,self.img_height,self.channels = input_shape[0],input_shape[1],input_shape[2] # image width,height,channels
    	img_width,img_height,channels = self.img_width,self.img_height,self.channels
    	img_size = img_width*img_height*channels
    	self.read_dense = Dense(5,activation="linear")
    	self.write_dense = Dense(5,activation="linear")
    	self.write_dense_image = Dense(self.write_window*self.write_window,activation="linear")
    	C = Input(shape=(img_width*img_height*channels,))
    	h_dec_prev = Input(shape=(h,))
    	X = Input(shape=(img_width*img_height*channels,))
    	Z = Lambda(self.Latent_Distribution_Sampling, output_shape=(self.h,))
    	R = Lambda(self.read)
    	W = Lambda(self.write)
    	Con = Lambda(self.concatenate)
    	attn = Lambda(self.attn_window)
    	self.mean = Dense(self.h,activation="linear")
    	self.log_var = Dense(self.h,activation="linear")
    	self.RNN_decoder = LSTM(self.h,return_state=True)
    	self.RNN_encoder = LSTM(self.h,return_state=True)
    	merge_add = keras.layers.Add()
    	merge_sub = keras.layers.Subtract()
    	C_out = Activation("linear")(C)
    	h_dec_prev_out = Activation("linear")(h_dec_prev)
    	for i in range(self.n):
    		C_sigma = Activation("sigmoid")(C_out)
    		X_hat = merge_sub([X,C_sigma])
    		attn_param = self.read_dense(h_dec_prev_out)
    		gx = Lambda(lambda x: (img_width+1)*(x[:,0]+1)/2)(attn_param)
    		gy = Lambda(lambda x: (img_height+1)*(x[:,1]+1)/2)(attn_param)
    		sigma2 = Lambda(lambda x: K.exp(x[:,2]))(attn_param)
    		delta = Lambda(lambda x: (max(img_width,img_height)-1)/(self.read_window-1)*K.exp(x[:,3]))(attn_param)
    		gamma = Lambda(lambda x: K.exp(x[:,4]))(attn_param)
    		Fx = Lambda(self.attn_window,arguments={"N":self.read_window,"dim":img_width})([gx,sigma2,delta])
    		Fy = Lambda(self.attn_window,arguments={"N":self.read_window,"dim":img_height})([gy,sigma2,delta])
    		r = Lambda(self.read,arguments={"N":self.read_window,"attention":self.attention,"A":img_width,"B":img_height})([X,X_hat,Fx,Fy,gamma])
    		h_dec = Con([r,h_dec_prev_out])
    		h_dec = keras.layers.Reshape((1,self.h+2*4))(h_dec)
    		if i==0:
    			h_enc_prev_out,enc_state_h, enc_state_c  = self.RNN_encoder(h_dec,initial_state=[h_dec_prev_out,h_dec_prev_out])
    		else:
    			h_enc_prev_out,enc_state_h, enc_state_c  = self.RNN_encoder(h_dec,initial_state=[enc_state_h,enc_state_c])
    		z_mean,z_log_var = self.mean(enc_state_h),self.log_var(enc_state_h)
    		z = Lambda(self.Latent_Distribution_Sampling,arguments={"h":self.h},output_shape=(self.h,))([z_mean, z_log_var])
    		if i==0:
    			kl = Lambda(lambda x:0.5*(K.square(x[0])+K.square(K.exp(x[1]))-2*x[1]-1))([z_mean,z_log_var])
    		else:
    			kl1 = Lambda(lambda x:0.5*(K.square(x[0])+K.square(K.exp(x[1]))-2*x[1]-1))([z_mean,z_log_var])
    			kl = merge_add([kl,kl1])
    		z = keras.layers.Reshape((1,self.h))(z)
    		if i==0:
    			h_dec_prev_out,dec_state_h,dec_state_c = self.RNN_decoder(z,initial_state=[h_dec_prev_out,h_dec_prev_out])
    		else:
    			h_dec_prev_out,dec_state_h,dec_state_c = self.RNN_decoder(z,initial_state=[dec_state_h,dec_state_c])
    		attn_param = self.write_dense(h_dec_prev_out)
    		gx = Lambda(lambda x: (img_width+1)*(x[:,0]+1)/2)(attn_param)
    		gy = Lambda(lambda x: (img_height+1)*(x[:,1]+1)/2)(attn_param)
    		sigma2 = Lambda(lambda x: K.exp(x[:,2]))(attn_param)
    		delta = Lambda(lambda x: (max(img_width,img_height)-1)/(self.write_window-1)*K.exp(x[:,3]))(attn_param)
    		gamma = Lambda(lambda x: K.exp(x[:,4]))(attn_param)
    		Fx = Lambda(self.attn_window,arguments={"N":self.write_window,"dim":img_width})([gx,sigma2,delta])
    		Fy = Lambda(self.attn_window,arguments={"N":self.write_window,"dim":img_height})([gy,sigma2,delta])
    		w = Lambda(self.write,arguments={"write_n":self.write_window,"attention":self.attention,"A":img_width,"B":img_height,"write_dense_image":self.write_dense_image})([Fx,Fy,gamma,dec_state_h])
    		C_out = merge_add([C_out,w])
    	C_out = Activation("sigmoid")(C_out)
    	Lx = K.mean(K.binary_crossentropy(X, C_out),axis=-1)
    	kl = K.mean(kl,axis=-1)
    	loss = Lx + kl
    	self.model = Model(input=[X,C,h_dec_prev],output=[C_out])
    	self.model.add_loss(loss)
    	self.model.compile(optimizer="rmsprop", loss=None)
    def fit(self,x,y,batch_size,nb_epoch,validation_data=None,shuffle=True):
        ''' call fit function
        Args:
            x (np.ndarray or [np.ndarray, np.ndarray])  : Input data for training
            y (np.ndarray)                              : Label data for training 
            samples_per_epoch (int)                     : Number of data for each epoch
            nb_epoch (int)                              : Number of epoches
            validation_data ((np.ndarray, np.ndarray))  : Validation data
            nb_val_samples (int)                        : Number of data yielded by validation generator
            shuffle (bool)                              : if True, shuffled randomly
        '''
        self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            callbacks=[self.tensorboard, self.checkpointer, self.earlystopping],
            validation_data=validation_data,
            shuffle=shuffle
        )

    def fit_generator(
        self,
        train_generator,
        samples_per_epoch,
        nb_epoch,
        validation_data=None,
        nb_val_samples=10000):
        ''' call fit_generator function
        Args:
            train_generator (object)        : image generator built by "build_generator" function
            samples_per_epoch (int)         : Number of data for each epoch
            nb_epoch (int)                  : Number of epoches
            validation_data (object/array)  : generator object or numpy.ndarray
            nb_val_samples (int)            : Number of data yielded by validation generator
        '''
        self.model.fit_generator(
            generator=train_generator,
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch,
            callbacks=[self.tensorboard, self.checkpointer, self.earlystopping],
            validation_data=validation_data,
            nb_val_samples=nb_val_samples
        )

    def build_predict(self):
    	print("make sure that the model is trained before prediction")
    	h = self.h
    	img_width,img_height,channels = self.img_width,self.img_height,self.channels
    	img_size = img_width*img_height*channels
    	Z = Lambda(self.Latent_Distribution_Sampling, output_shape=(self.h,))
    	R = Lambda(self.read)
    	W = Lambda(self.write)
    	Con = Lambda(self.concatenate)
    	attn = Lambda(self.attn_window)
    	mean = Dense(self.h,activation="linear")
    	log_var = Dense(self.h,activation="linear")
    	merge_add = keras.layers.Add()
    	merge_sub = keras.layers.Subtract()
    	h_dec_prev = Input(shape=(h,))
    	h_dec_prev_out = Activation("linear")(h_dec_prev)
    	out = {i:None for i in range(self.n)}
    	for i in range(self.n):
    		z = Lambda(self.Latent_Distribution_Sampling,arguments={"h":self.h},output_shape=(self.h,))([h_dec_prev_out, h_dec_prev_out])
    		if i==0:
                kl = Lambda(lambda x:0.5*(K.square(x[0])+K.square(K.exp(x[1]))-2*x[1]-1))([z_mean,z_log_var])
            else:
                kl1 = Lambda(lambda x:0.5*(K.square(x[0])+K.square(K.exp(x[1]))-2*x[1]-1))([z_mean,z_log_var])
                kl = merge_add([kl,kl1])
    		z = keras.layers.Reshape((1,self.h))(z)
    		if i==0:
    			h_dec_prev_out,dec_state_h,dec_state_c = self.RNN_decoder(z,initial_state=[h_dec_prev_out,h_dec_prev_out])
    		else:
    			h_dec_prev_out,dec_state_h,dec_state_c = self.RNN_decoder(z,initial_state=[dec_state_h,dec_state_c])
    		attn_param = self.write_dense(h_dec_prev_out)
    		gx = Lambda(lambda x: (img_width+1)*(x[:,0]+1)/2)(attn_param)
    		gy = Lambda(lambda x: (img_height+1)*(x[:,1]+1)/2)(attn_param)
    		sigma2 = Lambda(lambda x: K.exp(x[:,2]))(attn_param)
    		delta = Lambda(lambda x: (max(img_width,img_height)-1)/(self.write_window-1)*K.exp(x[:,3]))(attn_param)
    		gamma = Lambda(lambda x: K.exp(x[:,4]))(attn_param)
    		Fx = Lambda(self.attn_window,arguments={"N":self.write_window,"dim":img_width})([gx,sigma2,delta])
    		Fy = Lambda(self.attn_window,arguments={"N":self.write_window,"dim":img_height})([gy,sigma2,delta])
    		w = Lambda(self.write,arguments={"write_n":self.write_window,"attention":self.attention,"A":img_width,"B":img_height,"write_dense_image":self.write_dense_image})([Fx,Fy,gamma,dec_state_h])
    		if i==0:
    			C_out = Activation("linear")(w)
    		else:	
    			C_out = merge_add([C_out,w])
    		out[i] = Activation("sigmoid")(C_out)	
    	C_out = Activation("sigmoid")(C_out)
    	kl = K.mean(K.sum(kl,axis=1),axis=-1)
    	loss = kl
    	self.model_predict = Model(input=[h_dec_prev],output=[out[i] for i in range(self.n)])
    	self.model_predict.add_loss(loss)
    	self.model_predict.compile(optimizer="rmsprop", loss=None)


    def load_model(self, checkpoint_file):
        ''' restore model from checkpoint file (.hdf5) '''
        self.model = load_model(checkpoint_file)

    def export_to_json(self, save_root):
        ''' export model architecture config to json file '''
        with open(os.path.join(save_root, 'pixelcnn_model.json'), 'w') as f:
            f.write(self.model.to_json())

    def export_to_yaml(self, save_root):
        ''' export model architecture config to yaml file '''
        with open(os.path.join(save_root, 'pixelcnn_model.yml'), 'w') as f:
            f.write(self.model.to_yaml())


    @classmethod
    def predict(self, x, batch_size):
        ''' generate image pixel by pixel
        Args:
            x or [x,h] (x,h: numpy.ndarray : x = input image, h = latent vector
        Returns:
            predict (numpy.ndarray)        : generated image
        '''
        return self.model.predict(x, batch_size)
