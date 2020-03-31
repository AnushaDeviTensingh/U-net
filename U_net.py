import tensorflow as tf
import numpy as np
import h5py
import scipy.io as io
from tensorflow import keras
import os,sys
from tensorflow.keras import backend as bk
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

learning_rate=0.001
dropout=0.6
BN= True
istraining = True
image_size=[480,480,16]
conv_filters=[16,32,64,128,256]#[8,16,32,64,128]
kernal_size=[3,3,3]
# batch_size=5
train_path="/home/anu_1203/Server/Code/U_net/Training_data"
val_path="/home/anu_1203/Server/Code/U_net_keras/v4"
test_path="/home/anu_1203/Server/Code/U_net_keras/v4"
result_path="/home/anu_1203/Server/Code/U_net_keras/Results/Check_points/LR_0.001_DR_0.6_noBN/"
model_name="/home/anu_1203/Server/Code/U_net_keras/Results/Check_points/LR_0.001_DR_0.6_BN_model_epoch_11_loss_0.21.hdf5"
batch_size=1
epoch=50

file_name=os.listdir(train_path)
file_name_val=os.listdir(val_path)
file_name_test=os.listdir(test_path)

if not os.path.exists(result_path):
	os.mkdir(result_path)
	print(result_path + 'is created')
else:
	print('already exists')



class DataGen(keras.utils.Sequence):

	def __init__(self, ids, path, batch_size=batch_size,image_size=image_size[0]):
		self.ids=ids
		self.path=path
		self.batch_size=batch_size
		self.image_size=image_size
		self.on_epoch_end()
	
	def __load__(self, filename):
		# print(self.path,filename)
		directory=os.path.join(self.path, filename)
		f = h5py.File(directory) #for v7.3 mats
		value='train'
		images = np.array(f[value+'_patch'],dtype=np.float32)
		labels = np.array(f[value+'_labels'],dtype=np.float32)
		labels = labels[1,:,:,:]
		
		#images=np.expand_dims(images,axis=0)
		images=np.expand_dims(images,axis=3)
		images=np.transpose(images,(2,1,0,3))
	
		labels=np.expand_dims(labels,axis=3)
		labels=np.transpose(labels,(2,1,0,3))

		return images, labels
	def __getitem__(self, index):
		if(index+1)*self.batch_size>len(self.ids):
			self.batch_size=len(self.ids)-index*self.batch_size
		files_batch =self.ids[index*self.batch_size:(index+1)*self.batch_size]
		# print(files_batch)
	
		images=[]
		labels=[]
		
		for id_name in files_batch:
			# print(id_name)
			_img,_mask=self.__load__(id_name)
			images.append(_img)
			labels.append(_mask)
	
		images=np.array(images)
		labels=np.array(labels)

		return images, labels
	def on_epoch_end(self):
		pass
	def __len__(self):
		return int(np.ceil(len(self.ids)/float(self.batch_size)))
	

def down_blocks(x, filters, kernal_size, padding ="same", strides=1, BN=BN):
	c=keras.layers.Conv3D(filters, kernal_size, padding =padding, strides=strides)(x)
	if BN ==True:
		c=keras.layers.BatchNormalization()(c)
	c=keras.layers.LeakyReLU(alpha=0.01)(c)
	c=keras.layers.Dropout(rate=dropout)(c)
	c=keras.layers.Conv3D(filters, kernal_size, padding =padding, strides=strides)(c) #,activation="relu"
	if BN ==True:
		c=keras.layers.BatchNormalization()(c)
	c=keras.layers.LeakyReLU(alpha=0.01)(c)
	p=keras.layers.MaxPool3D(pool_size=(2, 2, 2))(c)
	return c,p

def up_block(x, skip, filters, kernal_size, padding= "same", strides=1, BN=BN):
	us=keras.layers.UpSampling3D(size=(2, 2, 2))(x)
	cat=keras.layers.Concatenate()([us,skip])
	c=keras.layers.Conv3D(filters, kernal_size, padding =padding, strides=strides)(cat)
	if BN ==True:
		c=keras.layers.BatchNormalization()(c)
	c=keras.layers.LeakyReLU(alpha=0.01)(c)
	c=keras.layers.Dropout(rate=dropout)(c)
	c=keras.layers.Conv3D(filters, kernal_size, padding =padding, strides=strides)(c)
	if BN ==True:
		c=keras.layers.BatchNormalization()(c)
	c=keras.layers.LeakyReLU(alpha=0.01)(c)
	return c

def bottle_neck(x,filters, kernal_size, padding= "same", strides=1,BN=BN):
	c=keras.layers.Conv3D(filters, kernal_size, padding =padding, strides=strides)(x)
	if BN ==True:
		c=keras.layers.BatchNormalization()(c)
	c=keras.layers.LeakyReLU(alpha=0.01)(c)
	c=keras.layers.Dropout(rate=dropout)(c)
	c=keras.layers.Conv3D(filters, kernal_size, padding =padding, strides=strides)(c)
	if BN ==True:
		c=keras.layers.BatchNormalization()(c)
	c=keras.layers.LeakyReLU(alpha=0.01)(c)
	c=keras.layers.Dropout(rate=dropout)(c)
	return c

############################################## U_net model ##############################################

def U_net():
	inputs=keras.layers.Input((image_size[0],image_size[1],image_size[2],1))
	#inputs=tf.expand_dims(inputs,axis=-1)
	#print(inputs)
	p0=inputs
	c1,p1=down_blocks(p0, conv_filters[0],kernal_size)
	c2,p2=down_blocks(p1, conv_filters[1],kernal_size)
	c3,p3=down_blocks(p2, conv_filters[2],kernal_size)
	c4,p4=down_blocks(p3, conv_filters[3],kernal_size)

	bn=bottle_neck(p4,conv_filters[4],kernal_size)

	u1=up_block(bn,c4,conv_filters[3],kernal_size)
	u2=up_block(u1,c3,conv_filters[2],kernal_size)
	u3=up_block(u2,c2,conv_filters[1],kernal_size)
	u4=up_block(u3,c1,conv_filters[0],kernal_size)

	outputs=keras.layers.Conv3D(1,[1,1,1], padding= "same", activation= "sigmoid")(u4)
	#print(outputs)    
	model =keras.models.Model(inputs,outputs)
	return model

############################### Dice loss #######################################
def dice_coef_loss(y_true,y_pred,smooth=1e-9):
	y_true_flat=bk.flatten(y_true)
	y_pred_flat=bk.flatten(y_pred)
	# y_true_flat=bk.flatten(bk.cast_to_floatx(y_true))
	# y_pred_flat=bk.flatten(bk.cast_to_floatx(y_pred))
	intersect=2.* (bk.sum(bk.abs(y_true_flat*y_pred_flat)))
	union=bk.sum(bk.square(y_true_flat))+bk.sum(bk.square(y_pred_flat))
	Dice=(intersect+smooth)/(union+smooth)
	return (1-Dice)

def dice(y_true,y_pred,smooth=1e-9):
	y_true_flat=bk.flatten(y_true)
	y_pred_flat=bk.flatten(y_pred)
	# y_true_flat=bk.flatten(bk.cast_to_floatx(y_true))
	# y_pred_flat=bk.flatten(bk.cast_to_floatx(y_pred))
	intersect=2.* (bk.sum(bk.abs(y_true_flat*y_pred_flat)))
	union=bk.sum(bk.square(y_true_flat))+bk.sum(bk.square(y_pred_flat))
	Dice=(intersect+smooth)/(union+smooth)
	return Dice

def exp_decay(epoch):
	initial_rate=learning_rate
	k=0.1
	lrate=initial_rate * np.exp(-k*epoch)
	print("Learning rate: ", lrate)
	return lrate


train_gen=DataGen(file_name,train_path,batch_size,image_size[0])
val_gen=DataGen(file_name_val,val_path,1,image_size[0])
test_gen=DataGen(file_name_test,test_path,1,image_size[0])
result_path=result_path + 'LR_0.001_DR_0.6_model_epoch_{epoch:02d}_loss_{loss:0.2f}_vloss_{val_loss:0.2f}.hdf5'

if istraining== True:
	model=U_net()
	# model.compile(optimizer="adam", loss='binary_crossentropy', metrics = ["acc"])
	# loss='mean_squared_error'
	optimizer= keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
	model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics = ["acc",dice])
	print(model.summary())
	train_step=len(file_name)//batch_size
	valid_step=len(file_name)//batch_size
	# print(train_step)
	# print(valid_step)
	lrs=LearningRateScheduler(exp_decay)
	checkpoint=ModelCheckpoint(result_path, monitor='val_loss',verbose=1,save_best_only=False, mode='min',save_weights_only= False, save_freq='epoch')
	callbacks_list=[lrs, checkpoint]
	# print(callbacks_list)
	model.fit(train_gen, validation_data=val_gen, epochs=epoch, steps_per_epoch=train_step, validation_steps=valid_step, callbacks=callbacks_list)
else:
	
	model=load_model(model_name, custom_objects={'dice_coef_loss':dice_coef_loss,'dice':dice})
	print(model.summary())
	j=0
	for i in file_name_val:
		print(i)
		x,y=val_gen.__getitem__(j)
		print(np.shape(x),np.shape(y))
		predicted=model.predict(x)
		result=model.evaluate(x,y)
		print(np.shape(predicted))
		print(np.shape(y))
		dice_pred=dice(y,predicted)
		sio.savemat(result_path+i,{'predicted_label':predicted})
		j=j+1
