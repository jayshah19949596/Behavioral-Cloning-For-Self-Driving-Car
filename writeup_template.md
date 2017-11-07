#  **Behavioral Cloning** 

##  Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## NVIDIA Model architecture
- Dropout were used to avoide overfitting

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 66x200x3 image   							    | 
| Convolution 5x5     	| 2x2 stride, 24 filters                       	|
| ELU					|												|
| Dropout				| 50 % keep probability 						|
| Convolution 5x5      	| 2x2 stride, 36 filters                       	|
| ELU           	    |                                               |
| Dropout				| 50 % keep probability 						|
| Convolution 5x5 		| 2x2 stride, 48 filters						|
| ELU                 	|                                			    |
| Dropout				| 50 % keep probability 						|
| Convolution 3x3 		| 2x2 stride, 64 filters						|
| ELU                 	|                                			    |
| Dropout				| 50 % keep probability 						|
| Convolution 3x3 		| 2x2 stride, 64 filters						|
| ELU                 	|                                			    |
| Dropout				| 50 % keep probability 						|
| Flatten            	|                                			    |
| Fully connected		| 100 nodes                      				|
| ELU                 	|                                			    |
| Dropout				| 50 % keep probability 						|
| Fully connected		| 50 nodes                      				|
| ELU                 	|                                			    |
| Dropout				| 50 % keep probability 						|
| Fully connected		| 10 nodes                      				|
| ELU                 	|                                			    |
| Fully connected		| 1 node                         				|


[image01]: ./writeup_images/NVIDIA_MODEL.PNG "NVIDIA"

![NVIDIA MODEL][image01]


You can refer the paper published by NIVIDIA :  [End to End Learning for Self-Driving Cars
](https://arxiv.org/pdf/1604.07316v1.pdf) 


## Loading Data

- We have driving_log.csv
- The csv has path to Left Image, Right Image, and Center Image and has angel measurement
- We use csv's paths to read the images


## Pre-Processing

- Cropped the Image appropriately 
- Applied Gaussian filter with filter size of 3*3
- Converted the Image from BGR to YUV space


## Correction Factor

- Correction factor used in the below cell is 0.2
- It is a parameter to tune
- For right camera images and left camera images the angel measurement is of center camera image which is wrong
- Correction factor helps us use the right camera images and left camera images effectively
- Correction factor makes the angel measurement to steer right by 0.2 for left images so that we can effectively use left images
- Correction factor makes the angel measurement to steer left by 0.2 for right images so that we can effectively use right images

## Flatten the Distribution

- This was a game changer for me 
- Somehow my model even after collecting the data was not performing good
- The model did drive with in the track but when ever there was a turn, it did not change the steering angle
- I saw the csv file and found that most of the steering angel data is 0 degree
- This is because the training track does not have much sharp turns
- Because of this somehow I felt that the training is biased
- We have told the network for most of the data to keep strring angel to 0
- Because of this I think my model was more inclined to go straight even when there was a turn 
- I started for looking for solutions online
- My approach and code for the below cell is heavily inspired by https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project
- I decided to throw away some of the data looking at above link's solution
- Calculated **`avg_samples_per_bin`** 
- all the steering angle data for a particular bin that had number of data < **`avg_samples_per_bin`** were kept
- The bins that had data more then **`avg_samples_per_bin`** were kept with a **`1-keep_probability`**
- Somehow with this solution the model performed good but got confused at the bridge
- So no data from data_3 folder or data_4 folder was thrown away because those folder had training data for bridge

### Data Distribution of steering angles before Flattening is shown below
[image02]: ./writeup_images/Before_Flattening.PNG "Before_Flattening"
![Before_Flattening][image02]

### Data Distribution of steering angles after Flattening is shown below
[image03]: ./writeup_images/After_Flattening.PNG "After_Flattening"
![After_Flattening][image03]


## Data Augmentation

- I flipped the images horizontally
- The angles of the flipped images were multiplied by -1.0
- This helped in increasing the amount of data 
- It also helps in model to generalize on the training track 
- This Project made one thing clear that if you have data than only your model can shine and achieve the goal


## GENERATORS

- The images captured in the car simulator are much larger than the images encountered in the Traffic Sign Classifier Project, a size of 160 x 320 x 3 
- 10,000 simulator images would take over 1.5 GB
- That's a lot of memory
- preprocessing data can change data types from an `int` to a `float`, which can increase the size of the data by a factor of 4.
- Generators can be a great way to work with large amounts of data.
- Instead of storing the preprocessed data in memory all at once, using a generator you can pull pieces of the data and process them on the fly only when you need them.
- This is much more memory-efficient.
- To return generator instead of using **`rerturn`** keyword we use **`yield`** keyword
- **`yield`** keyword is at line 34 of below cell
- Keras has inbuilt function **`fit_generator`**:
   * fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
   
   
- Arguments:
   * generator: A generator or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing.
   * steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch
   * epochs: Integer, total number of iterations on the data.
   * verbose: Verbosity mode, 0, 1, or 2.
   * callbacks: List of callbacks to be called during training.
   * validation_data: This can be either a generator for the validation data or tuple (inputs, targets) or a tuple (inputs, targets, sample_weights)
   * validation_steps: Only relevant if `validation_data` is a generator
   * class_weight: Dictionary mapping class indices to a weight for the class.
   * max_queue_size: Maximum size for the generator queue
   * workers: Maximum number of processes to spin up when using process based t
   * use_multiprocessing: If True, use process based threading
   * shuffle: Whether to shuffle the order of the batches at the beginning of each epoch
   * initial_epoch: Epoch at which to start training
   
   
- Return :
    * A History object.


## Final Output

[image04]: ./writeup_images/FinalOP.PNG "Final_op"
![Final_op][image04]


