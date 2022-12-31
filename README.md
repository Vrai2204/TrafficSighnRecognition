# TrafficSighnRecognition


**About the Dataset** :

For this project, we are using the German Traffic Sign Recognition Benchmark dataset that contains a variety of traffic signs.
It is further classified into different classes. The dataset will be quite varying, some of the classes may have many images 
while some classes have few images. The dataset will have a train folder that contains images inside each class and a test 
folder that we will use for testing our model.

**Approach** :

Initially, the 'train' folder contains different folders each representing a different class and each class has various 
images of pictures all belonging to the same traffic sign label. Each train folder represents a traffic sign labeled. 
It can be e.g., a stop sign, yield, 30 km/h speed limit, etc.

**Additional Implementation:**

- We have added additional test datasets to the original one where we performed the preprocessing, splitting, testing, and validation of these datasets.
- We have integrated codes from multiple references and websites in our model preparation where we made use of additional Keras API layers.
- We tried implementing 2 models for our project, **pytorch** model and the **sequential**** CNN** model.
- A sequential CNN model includes different layers where each layer forms the entire network using exactly one input and output which is stacked together.
- Pytorch is used for deep learning and research due to its speed and its benefits in the process between research prototyping and deployment.
- We are implementing a basic GUI (Graphical User Interface) to verify the classification of images, where we can able to upload an image of a traffic sign to the interface and it will show the classification (Showing the name of the traffic sign) of image.

**Model Implementation** :

We used a convolutional neural network (CNN) which is a type of artificial neural network used in image recognition and processing
that is specifically designed to process pixel data.

After splitting the data into training datasets, we build a CNN model by testing it on the training dataset for the 
classification of images in the dataset into their respective categories.

For the CNN model architecture, we used below Keras API layers:

1. Convolution Layer – It makes the network learns filters that activate when it detects a specific feature type.
2. Pooling Layer - It [partitions](https://en.wikipedia.org/wiki/Partition_of_a_set) the input image into a set of rectangles and, for each such sub-region, outputs the maximum.
3. Normalization Layer – It will normalize each of the inputs in the batch independently across all features.
4. Reshaping Layer - The Reshape layer can be used to change the input dimensions, without changing its data. Just like the Flatten layer, only it will change the dimensions; no data is copied in the process.
5. Core Layer – It will have layers like the Dense layer, Activation Layer, Embedding layer, Masking layer, and Lambda layer.
6. Regularization layer - Regularization is a set of techniques that can prevent overfitting in neural networks and thus improve the accuracy of a Deep Learning model when facing completely new data from the problem domain.

<img width="426" alt="image" src="https://user-images.githubusercontent.com/92072470/210151043-3475eaa0-3d45-42a5-95aa-9eae05999464.png">
<img width="495" alt="image" src="https://user-images.githubusercontent.com/92072470/210152210-6b0e3bca-0371-4e07-9554-efd163d996c0.png">


**Model Optimization:**

- We used Adam for optimization to update network weights iterative based on the training data.
- This procedure is used when there is a problem with large data involved and can make it work efficiently.
- Adam optimizer gives higher performance when compared to the other algorithms.
- We used Epochs to define the number of times that the learning algorithm will work through the entire training dataset.
- An epoch in machine learning means one complete pass of the training dataset through the algorithm. This epochs number is an 
  important hyperparameter for the algorithm.
- It specifies the number of epochs or complete passes of the entire training dataset passing through the training or learning process of the algorithm.

  With each epoch, the dataset's internal model parameters are updated.
<img width="468" alt="image" src="https://user-images.githubusercontent.com/92072470/210152215-dcc32cfd-e532-40eb-8bac-37b29baf9d62.png">

**Data Augmentation and Model Fitting:**

- For the data augmentation, we used ImageDataGenerator class which can apply transformations randomly on the image data for 
  each individual image by running it and passing it to the model. So, every time the model runs it will have a different image 
  generated for a single type of image in order to identify by the model.
- Then model.fit() is used to fit the model on batches with real-time data augmentation and for the batch size configuration, 
  and data generator preparation to get batches of images we used the flow() function.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/92072470/210152229-05572a99-e815-4239-8de3-39cb1ee54de0.png">

**Model Accuracy:**

After augmenting and training the model, evaluation of our model is done and we loaded the test data into the model and performed 
resizing of those images from the test dataset. Then we ran the predictions on the test data to test our 
accuracy score of 92.92%.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/92072470/210152243-06587d6c-b863-40ed-8bc5-7b5e1c829ece.png">

<img width="468" alt="image" src="https://user-images.githubusercontent.com/92072470/210152251-7a653c5a-1a92-49ba-b138-71a2638a970e.png">

**Confusion Matrix:**

We have plotted a figure for the confusion matrix using the classes dictionary where the description of each class is stored in that dictionary for the classification of data. Below is the figure.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/92072470/210152256-0ac3f02b-0302-48fd-8582-166f4efe0654.png">

**Classification Report:**

<img width="468" alt="image" src="https://user-images.githubusercontent.com/92072470/210152261-bab48cd0-fbdc-4e25-b70e-477106d24a4b.png">

**GUI:**

A graphical user interface (GUI) is helpful in testing and seeing the results of our model prediction and saves a lot of time. We used Tkinter to make a graphical user interface which is an inbuilt library of python.

From the interface of the GUI application, we can upload an image and extract the file path of the image. Then we use the trained model that will take the image data as input and displays the class to which it belongs. We then used the [**dictionary**](https://data-flair.training/blogs/python-dictionary/) to see the name of the class. We created a new python file and named it as traffic\_sign\_gui.py.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/92072470/210152266-d3cbd938-a280-4ced-b693-c64deadf1085.png">

**References:**

1. A real-time trafﬁc sign recognition system- [https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-35-11-1907&id=399639](https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-35-11-1907&id=399639)
2. https://debuggercafe.com/traffic-sign-recognition-using-pytorch-and-deep-learning/
3. Research on Traffic Sign Detection Based on Convolutional Neural Network-[https://dl.acm.org/doi/abs/10.1145/3356422.3356457](https://dl.acm.org/doi/abs/10.1145/3356422.3356457)
4. A system for traffic sign detection, tracking, and recognition using color, shape, and motion information- [https://www.researchgate.net/publication/224255280\_Real-time\_traffic\_sign\_recognition\_system](https://www.researchgate.net/publication/224255280_Real-time_traffic_sign_recognition_system)
5. color vision model-based approach for segmentation of trafﬁc signs- [https://www.researchgate.net/publication/337800234\_Traffic\_Sign\_Recognition\_with\_a\_small\_convolutional\_neural\_network](https://www.researchgate.net/publication/337800234_Traffic_Sign_Recognition_with_a_small_convolutional_neural_network)
6. Enhancement for road sign images and its performance evaluation- [https://www.diva-portal.org/smash/get/diva2:523372/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:523372/FULLTEXT01.pdf)
7. [https://medium.com/dataflair/class-data-science-project-for-2020-traffic-signs-recognition-12b09c131742](https://medium.com/dataflair/class-data-science-project-for-2020-traffic-signs-recognition-12b09c131742)
 
