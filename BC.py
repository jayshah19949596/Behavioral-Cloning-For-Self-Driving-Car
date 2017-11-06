import csv
import cv2
import numpy as np
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Conv2D, ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.regularizers import l2


def plot_loss_graph(history_object):
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()


def my_model(train_generator, validation_generator, no_train_data, no_of_validation_data):
	model = Sequential()
	
	model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
	
	model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(64, 3, 3, activation='elu'))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(64, 3, 3, activation='elu'))
	model.add(Dropout(0.5))
	
	model.add(Flatten())
	
	model.add(Dense(100, activation='elu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(50, activation='elu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(10, activation='elu'))
	
	model.add(Dense(1))
	
	model.compile(loss="mse", optimizer="adam")
	
	history_object = model.fit_generator(train_generator,
										 samples_per_epoch=no_train_data,
										 validation_data=validation_generator,
										 nb_val_samples=no_of_validation_data,
										 nb_epoch=3,
										 verbose=1)
	
	# plot_loss_graph(history_object)
	
	model.save("more_data.h5")
	
	# print the keys contained in the history object
	print(history_object.history.keys())
	
	# plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()


def perform_data_augmentation(images, measurements):
	augmented_images = []
	augmented_measurements = []
	
	for image, measurement in zip(images, measurements):
		augmented_images.append(image)
		augmented_measurements.append(measurement)
		augmented_images.append(cv2.flip(image, 1))
		augmented_measurements.append(measurement * -1.0)
	
	return np.array(augmented_images), np.array(augmented_measurements)


def pre_process_image(img):
	"""
    Method for pre-processing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
    received in RGB)
    """
	# original shape: 160x320x3, input shape for neural net: 66x200x3
	new_img = img[50:140, :, :]
	# apply subtle blur
	new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
	new_img = cv2.resize(new_img, (200, 66), interpolation = cv2.INTER_AREA)  # scale to 66x200x3 (same as NVIDIA)
	new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
	return new_img


def random_distort(img, angle):
	"""
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position """
	new_img = img.astype(float)
	# random brightness - the mask bit keeps values from going beyond (0,255)
	value = np.random.randint(-28, 28)
	if value > 0:
		mask = (new_img[:, :, 0] + value) > 255
	if value <= 0:
		mask = (new_img[:, :, 0] + value) < 0
	new_img[:, :, 0] += np.where(mask, 0, value)
	# random shadow - full height, random left/right side, random darkening
	h, w = new_img.shape[0:2]
	mid = np.random.randint(0, w)
	factor = np.random.uniform(0.6, 0.8)
	if np.random.rand() > .5:
		new_img[:, 0:mid, 0] *= factor
	else:
		new_img[:, mid:w, 0] *= factor
	# randomly shift horizon
	h, w, _ = new_img.shape
	horizon = 2*h/5
	v_shift = np.random.randint(-h/8, h/8)
	pts1 = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
	pts2 = np.float32([[0, horizon+v_shift], [w, horizon+v_shift], [0, h], [w, h]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	new_img = cv2.warpPerspective(new_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
	return new_img.astype(np.uint8), angle


def read_data(files):
	samples = []
	angles = []
	image_paths = []
	for file in files:
		with open(file) as csvfile:
			reader = csv.reader(csvfile)
			i = 0
			for line in reader:
				if i == 0:
					i += 1
					continue
				line.append(file.split("/")[0])
				
				for i in range(0, 3):
					angle = get_measurement(line, i)
					angles.append(angle)
					samples.append(file.split("/")[0])
					
					if line[-1] == "data":
						split_values = line[i].split("/")
						source_path = line[-1] + "/" + "IMG" + "/" + split_values[-1]
					# print(source_path)
					
					elif line[-1] == "data_1" or line[-1] == "data_2" or line[-1] == "data_3":
						split_values = line[i].split("\\")
						source_path = line[-1] + "/" + split_values[-2] + "/" + split_values[-1]
					
					image_paths.append(source_path)
	
	return samples, angles, image_paths


def data_visualization(angles):
	num_bins = 23
	avg_samples_per_bin = len(angles) / num_bins
	hist, bins = np.histogram(angles, num_bins)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center', width=width)
	plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
	plt.show()


def flatten_distribution(samples, angles, image_paths):
	keep_probs = []
	num_bins = 23
	avg_samples_per_bin = len(angles) / num_bins
	hist, bins = np.histogram(angles, num_bins)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	target = c * .5
	for i in range(num_bins):
		if hist[i] < target:
			keep_probs.append(1.)
		else:
			keep_probs.append(1. / (hist[i] / target))
	remove_list = []
	for i in range(len(angles)):
		for j in range(num_bins):
			if bins[j] < angles[i] <= bins[j + 1] and samples[i] != "data_4" and samples[i] != "data_3":
				# delete from X and y with probability 1 - keep_probs[j]
				if np.random.rand() > keep_probs[j]:
					remove_list.append(i)
	
	image_paths = np.delete(image_paths, remove_list, axis=0)
	angles = np.delete(angles, remove_list)
	
	# print histogram again to show more even distribution of steering angles
	hist, bins = np.histogram(angles, num_bins)
	plt.bar(center, hist, align='center', width=width)
	plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
	plt.show()
	
	image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles,
																					  test_size=0.05,
																					  random_state=42)
	return image_paths_train, image_paths_test, angles_train, angles_test


def prepare_generators(img_paths, angles, batch_size):
	num_samples = len(img_paths)
	while 1:  # Loop forever so the generator never terminates
		
		for offset in range(0, num_samples, batch_size):
			
			batch_angles = angles[offset:offset + batch_size]
			batch_img_paths = img_paths[offset:offset + batch_size]
			
			images = []
			measurements = []
			
			for x in range(0, len(batch_angles)):
				image = cv2.imread(batch_img_paths[x])
				if image is None:
					# print(None)
					continue
				# else:
				# 	print(image.shape)
				img, angle = random_distort(pre_process_image(image), batch_angles[x])
				
				images.append(img)
				measurements.append(angle)
			
			images, measurements = perform_data_augmentation(images, measurements)
			
			yield shuffle(np.array(images), np.array(measurements))


def get_generators():
	file_1 = "data/driving_log.csv"
	file_2 = "data_1/driving_log.csv"
	file_3 = "data_2/driving_log.csv"
	file_4 = "data_3/driving_log.csv"
	file_5 = "data_4/driving_log.csv"
	
	samples, angles, image_paths = read_data([file_1, file_2, file_3, file_4, file_5])
	
	data_visualization(angles)
	image_paths_train, image_paths_test, angles_train, angles_test = flatten_distribution(samples, angles, image_paths)
	
	# compile and train the model using the generator function
	train_generator = prepare_generators(image_paths_train, angles_train, batch_size=32)
	validation_generator = prepare_generators(image_paths_test, angles_test, batch_size=32)
	
	return train_generator, validation_generator, len(image_paths_train), len(image_paths_test)


def prepare_data():
	samples = read_data()
	images = []
	measurements = []
	for sample in samples:
		for i in range(3):
			source_path = "data/" + sample[i]
			image = cv2.imread(source_path)
			if image is None:
				continue
			images.append(image)
			measurement = get_measurement(sample, i)
			measurements.append(measurement)
	
	images, measurements = perform_data_augmentation(images, measurements)
	return images, measurements
	
	
	# X_train, y_train = prepare_data()
	# print(X_train.shape)
	# nvidia(X_train, y_train)
	
# nvidia_gen(train_gen, validation_gen, no_train_samples, no_of_validation_samples)
INPUT_SHAPE = (160, 320, 3)
train_gen, validation_gen, no_train_samples, no_of_validation_samples = get_generators()
my_model(train_gen, validation_gen, no_train_samples, no_of_validation_samples)

# 10,365 Files
# 161 mb on disk
