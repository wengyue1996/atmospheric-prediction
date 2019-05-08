import tensorflow as tf
import model
import numpy as np
import os
import file_reader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def batch_data(images):
	input_batch = np.array([[[image]] for image in images[:-1]])
	input_batch = np.transpose(input_batch, (1,0,3,4,2))
	output_batch = np.array([[[image]] for image in images[1:]])
	output_batch = np.transpose(output_batch, (1,0,3,4,2))
	return input_batch, output_batch



data_directory = '/media/ian/B2D61116D610DC831/8-7-18 Wavefronts/'
files = os.listdir(data_directory)

def segment_files(filelist):
	train_filelist, test_filelist = [], []
	for file in filelist:
		num = np.random.random()
		if(num>0.2):
			train_filelist.append(file)
		else:
			test_filelist.append(file)
	return train_filelist, test_filelist

train_filelist, test_filelist = segment_files(files)


convlstm_model = model.get_model()
convlstm_model.compile(
	optimizer=tf.train.AdamOptimizer(0.001),
	loss='mse',
	metrics=['mse'])

num_epochs = 100

for epoch in range(num_epochs):
	print('Epoch {}'.format(epoch))
	for file in train_filelist[:10]:
		if(file.startswith('wavefront')):
			input_batch, output_batch = batch_data(file_reader.read_file(data_directory+file))
			convlstm_model.train_on_batch(input_batch, output_batch)
	average_error = 0
	for file in test_filelist[:10]:
		input_batch, output_batch = batch_data(file_reader.read_file(data_directory+file))
		error = convlstm_model.test_on_batch(input_batch, output_batch)
		print(error)
		average_error += error
	average_error = average_error/len(test_filelist)
	print('Average error: {}'.format(average_error))
