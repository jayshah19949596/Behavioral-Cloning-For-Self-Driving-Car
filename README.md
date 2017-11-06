#**Behavioral Cloning** 

##Writeup Template

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

- Corection factor used in the below cell is 0.2
- It is a parameter to tune
- For right camera images and left camera images the angel measurement is of center camera image which is wrong
- Corection factor helps us use the right camera images and left camers images effectively
- Correction factor makes the angel measurement to stir right by 0.2 for left images so that we can effectively use left images
- Correction factor makes the angel measurement to stir left by 0.2 for right images so that we can effectively use right images

## Flatten the Distribution

- This was a game changer for me 
- Somehow my model even after collecting the data was not performing good
- The model did drive with in the track but when ever there was a turn, it did not change the stirring angle
- I saw the csv file and found that most of the stirring angel data is 0 degree
- This is because the training trach does not have much sharp turns
- Because of this somehow I felt that the training is biased
- We have told the network for most of the data to keep strring angel to 0
- Because of this I think my model was more inclined to go straight even when there was a turn 
- I started for looking for solutions online
- My approach and code for the below cell is heavily inspired by https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project
- I decided to throw away some of the data looking at above link's solution
- Calculated **`avg_samples_per_bin`** 
- all the stirring angle data for a particular bin that had number of data < **`avg_samples_per_bin`** were kept
- The bins that had data more then **`avg_samples_per_bin`** were kept with a **`1-keep_probability`**
- Somehow with this solution the model performed good but got confused at the bridge
- So no data from data_3 folder or data_4 folder was thrown away because those folder had training data for bridge

### Data Distribution of stirring angles before Flattening is shown below
[image02]: ./writeup_images/Before_Flattening.PNG "Before_Flattening"
![Before_Flattening][image02]

### Data Distribution of stirring angles after Flattening is shown below
[image03]: ./writeup_images/After_Flattening.PNG "After_Flattening"
![After_Flattening][image03]


## Data Augmentation

- I flipped the images horizontally
- The angles of the flipped images were multiplied by -1.0
- This helped in increasing the amount of data 
- It also helps in model to generalize on the training track 
- This Project made one thing clear that if you have data than only your model can shine and achieve the goal

## Final Output

[image04]: ./writeup_images/Final_op.PNG "Final_op"
![Final_op][image04]


