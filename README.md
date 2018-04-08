# **Traffic Sign Classifier Project**
---
**Build a Traffic Sign Recognition Project**

**In this project I have worked on a deep learning model based on LeNet architecture by Yan LeCun to classify german traffic signs using the dataset [German Traffic Signs Dataset "GTSRB"](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) .**
**Then I have tested the model on seven german traffic sign images downloaded from the internet.**

**In the following  writeup file, I will be explaining each part of the project**

## Project Pipeline Stepts:
---
**Step 0:** Load The data

**Step 1:** Dataset Summary & Exploration
  * Provide a basic summary of the dataset
  * Explatory visualization
    * Plotting 50 random samples of the dataset with their class id (label)
    * Plotting the histogram of classes and their frequencies in the dataset
    
**Step 2:** Design and Test a Model Architeture
  * Pre-processing of the dataset
    * Grayscaling
    * Normalization
    * Visualization of pre-processed data
  * Shuffle the dataset
  * Model Architecure
    * Setting the `EPOCHS` and `BATCH_SIZE`
    * Make use of the LeNet Architecture **with adding dropout for the fully connected layers**
    * Setting the features, labels, and dropout probability as tensorflow placeholders
    * Setting the learning rate, calculating cross-entropy, minimizing cross-entropy
    * Training the model
    * Evaluate the model using its validation set
  * After tweaking the whole hyper-parameter and taking the decision, testing the model using the test set
    
**Step 3:** Test the Model on New Images
  * Load the test images
  * Test images summary
  * Visualize the test images
  * Pre-process the test images
    * Grayscaling
    * Normalization
    * Visualization of pre-processed data
    * Test the model on the new images
  * Output the Top 5 Softmax Probabilities for Each Image of Test Images
  * Visualizing the predicted images accuracies in connection with the original test images

## Environment:
---
* AWS carnd Instance
* Python 3.6.4
* Anaconda 4.4.10

## Step 0: Load The Data
---
I have loaded the dataset from the pickle files provided. The dataset is divided into training set, validation set, and test set. Each set has its corresponding ground-truth labels. Here is the code for this step:
```python
# Load pickled data
import pickle
training_file   = 'train.p'
validation_file = 'valid.p'
testing_file    = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

## Step 1: Dataset Summary & Exploration
---
### Provide a Basic Summary of the Dataset
I used the len(), max(), and the shape attribute of the dataset arrays to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410 
* The size of test set is 34799
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

### Include an exploratory visualization of the dataset
* Plotting 50 random samples of the dataset with their class id (label)
I have plotted 50 reandom samples of the dataset with their class id (label) above it as a title. This gives some intuition about a small sample of the dataset.
Here is how the visualization will look like - check the complete picture at my IPyhton notebook:
![data_vis](https://i.imgur.com/eYYx98V.png)

* Plotting the histogram of classes and their frequencies in the dataset
I have plotted a histogram for the classes ids (labels) frequencies through our dataset (i.e., how many times a definite class exists in our dataset). 
This step is useful to check the distribution of our classes. Our model may be biased to some classes of higher frequence. This step _may_ lead to data augmentation if some of the classes exist in very low frequecny comparing to other ones.
Here is how the histogram of classes ids frequency:
![hist](https://i.imgur.com/urACCAY.png)
> It is worth to mention that the histogram really shows low frequenceis for some classses but I have delayed the step of data augmentation till checking the accuracy for my model with the dataset as it is. **I will be commenting on this point in the conclusion.**

## Step 2: Design and Test a Model Architecture
---
Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
### Pre-processing of the dataset
  * Grayscaling
Grayscaling the dataset (Training set, Validation set, and Test set). In somecases, it is more convientient for the model to feed single channel (grayscaled) images rather than 3-channels (Colored, e.g. RGB) images.
It is worth to mention that whenever using the openCV function cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) you have to take care of two things: 1) whether the images are read in RGB or BGR or even any other color space. 2) The cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) return an image without the channel dimension. So, for our model, we will need this dimension, so you have to reshape the image to the proper dimension.
That is why I prefered to use the following way for grayscaling using NumPy, for each image, divide channel pixels by 3, then summing them together with keeping the dimension of the number of channels.
The following is the code for grayscaling:
```python
# Convert X_train, X_valid, and X_test to grayscale
gray_scale_X_train = np.sum(X_train/3, axis=3, keepdims=True)
gray_scale_X_valid = np.sum(X_valid/3, axis=3, keepdims=True)
gray_scale_X_test = np.sum(X_test/3, axis=3, keepdims=True)
```
Visualizing 32 random data points in RGB and its correspoding grayscale image beside it (i.e., total 64 images).
Here is the visualization will look like - check the complete picture at my IPyhton notebook:
![gry_scale_vis](https://i.imgur.com/SV9hlA2.png)
  * Normalization
The dataset (Training set, Validation set, and Test set) are normalized so that the data has mean zero and equal variance. For image data, 
`(pixel - 128)/ 128`

  * Visualization of processed data
Visualizing 32 random data points in grayscale and its correspoding normalized image beside it (i.e., total 64 images).
Here is the visualization will look like - check the complete picture at my IPyhton notebook:
![pre-proc-visual](https://i.imgur.com/2Eh9kgV.png)

### Shuffle the dataset
Here, we shuffle the training set in order not to make the model biased by the order of images
```python
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
```
### Model Architecure
* Setting the `EPOCHS` and `BATCH_SIZE`
  * The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy. 
  * Here I have chosen the `EPOCHS` to be **100** and the `BATCH_SIZE` to be **128**

* Make use of the LeNet Architecture **with adding dropout for the fully connected layers**
  * I used the LeNet Architecture by Yan LeCun that is shown below as a base **with adding dropout for the fully connected layers**. ![LeNet Architecture](https://i.imgur.com/98tBUEC.png)
  *  **Dropout** is used in order to prevent our model to memorize the training set. We get consensus opinion by averaging the activations (I used a dropout of 0.5).
  * The following table describes the model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Layer 1: Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|	Activation function											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Layer 2: Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|	Activation function											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Layer 3: Fully connected		| output = 120        									|
| RELU					|	Activation function											|
| Dropout					|	keep_prob = 0.5										|
| Layer 4: Fully connected		| output = 84        									|
| RELU					|	Activation function											|
| Dropout					|	keep_prob = 0.5										|
| Layer 5: Fully connected		| output = 43        									|
  * The following is the python code for the model:
```python
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') #+ conv1_b
    conv1   = tf.nn.bias_add(conv1, conv1_b)

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID')#+ conv2_b
    conv2   = tf.nn.bias_add(conv2, conv2_b)
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.add(tf.matmul(fc0, fc1_W) , fc1_b)
    
    # Activation.
    fc1    = tf.nn.relu(fc1)
    
    # Dropout.
    fc1    = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.add(tf.matmul(fc1, fc2_W) , fc2_b)
    
    # Activation.
    fc2    = tf.nn.relu(fc2)
    
    # Dropout.
    fc2    = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc2, fc3_W) , fc3_b)
    
    return logits
```

* Setting the features, labels, and dropout probabilty as tensorflow placeholders
  * `x` is a placeholder for a batch of input images.
  * `y` is a placeholder for a batch of output labels.
  * `one_hot_y` is a placeholder for labels in one-hot-encoded format
  * `keep_prob` is a placeholder for keeping the number of activations

* Setting the learning rate, calculating cross-entropy, minimizing cross-entropy
  * Here, I used learning `rate` of 0.001. Cross-entropy for our model is caluclated using `softmax_cross_entropy_with_logits`. 
  * Then cross-entropy is optimized using Adam Optimizer `tf.train.AdamOptimizer`

* Training the model
  * Run the training data through the training pipeline to train the model.
  * Before each epoch, shuffle the training set.
  * After each epoch, measure the loss and accuracy of the validation set.
  * A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation sets implies underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.
  * Save the model after training.
  
* Evaluate the model using its validation set
  * After tweaking the whole hyper-parameter and taking the decision for the model, testing the model using the test set

* My final model results were:
  * validation set accuracy of 97 %
  * test set accuracy of 95 %

> Actually, this was initial thinking to tackle the traffic signs classification problem in which it gives outstanding accuracies for both validation set and test set.
## Step 3: Test a Model on New Images
---
### Testing our model on seven images for german traffic signs
#### Load and output the images
Here, I have loaded the downloaded test images
```python
images_list = os.listdir("Test_Images_From_Internet/")
print(images_list)
test_images = []
for img in images_list:
    if not img.startswith('.'): # To ignore the hidden files
        image = cv2.imread("Test_Images_From_Internet/"+img)
        image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_images.append(image)
        
test_images = np.asarray(test_images)
```

#### Test Images summary
I printed a summary for the test images
  * Test Images Set Shape: 7
  * Test Image Shape: 32 x 32 x 3
  * Number of Exaples or Classes in Test Images Set: 7
  
#### Visualize the test images
Here are the test images:
![test_images](https://i.imgur.com/I2bkuvB.png)

#### Pre-process the test images
  * Grayscaling
  As we have done with the dataset, I have converted that test images to grayscale
  ![test_images_grayscale](https://i.imgur.com/Vetkt79.png)
  * Normalization
  Then I have normoalized the test images (grayscale on the left and normalized on the right)
  ![test_images_norm](https://i.imgur.com/OPE5z56.png)

#### Test the model on the new images
The test images are predicted using our model
```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(test_data, test_labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution     		| General caution  		| 
| Speed limit (30km/h)     			| Speed limit (30km/h)			|
| Priority road					| Priority road			|
| Turn left ahead	      		| Turn left ahead		|
| Right-of-way at the next intersection			| Right-of-way at the next intersection |
| Keep right			| Keep right |
| Road work		| Road work |

**Accuracy is 100 %**

#### Output the Top 5 Softmax Probabilities for Each Image of Test Images
  * I have used the `tf.nn.top_k` to print the top 5 softmax probabilites of the test images.
  * In the following, I am listing the top first probablitity of each test image
  
| Probability			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution     		| 100 %  		| 
| Speed limit (30km/h)     			| 99 %			|
| Priority road					| 100 %			|
| Turn left ahead	      		| 100 %		|
| Right-of-way at the next intersection			| 100 % |
| Keep right			| 100 % |
| Road work		| 100 % |


#### Visualizing the predicted images accuracies in connection with the original test images
The following is a visualization of the top 5 probabilites with the test images and predicted image
![top_five_vis](https://i.imgur.com/W45tTe2.png)

## Conclusion
---
  * Using the LeNet Architecture with adding dropout gives outstanding validation set accuracy and test set accuracy.
  * The histogram for the dataset classes frequency gives an indication that the classes are not well-distributed. This leads to data augmentation to have a better distribution for the classes through our dataset. However, I have chosen to continue for my initial thinking to check the mentioned model above with the dataset withouth augmentation. However, data augmentation will increase both validation accuracy and test accuracy for more than 97 % and 95 %.
  * The [published baseline model for traffic signs classification problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Pierre Sermanet and Yann LeCun is another good improvent which will enhance the validation and test sets accuracies.
  * For testing the model with internet-downloaded images, the model gives high accuracies when the images are well-scaled and well-resized. Even though, when the test images are distored, the model will fail to classify some images
