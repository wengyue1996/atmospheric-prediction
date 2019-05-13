import tensorflow as tf
import numpy as np

data_format = 'channels_last'
layer1_filters = 10
layer2_filters = 10
convlstm_filters = 10
layer3_filters = 5
layer4_filters = 1

def get_model():

	layer1= tf.keras.layers.Conv3D(input_shape=(None, None, None, 1), filters=layer1_filters, kernel_size=(1,3,3), padding='same', data_format=data_format)
	layer2= tf.keras.layers.Conv3D(filters=layer2_filters, kernel_size=(1,3,3), padding='same', data_format=data_format, activation=tf.tanh)
	convlstm_unit1 = tf.keras.layers.ConvLSTM2D(filters=convlstm_filters, kernel_size=(3,3), padding='same', data_format=data_format, return_sequences=True)
	convlstm_unit2 = tf.keras.layers.ConvLSTM2D(filters=convlstm_filters, kernel_size=(5,5), padding='same', data_format=data_format, return_sequences=True)
	layer3= tf.keras.layers.Conv3D(filters=layer3_filters, kernel_size=(1,3,3), padding='same', data_format=data_format, activation=tf.tanh)
	layer4= tf.keras.layers.Conv3D(filters=layer4_filters, kernel_size=(1,3,3), padding='same', data_format=data_format)

	model = tf.keras.Sequential([layer1, layer2, convlstm_unit1, convlstm_unit2, layer3, layer4])

	return model