# Age, Gender, and Race Classification with Multi-outputs Convolutional Neural Networks 
This project built a multi-output deep convolutional neural to classify the age, gender, and race for each image included in the UTK Face dataset, reaching an accuracy of 91.22% for gender and 81.23% for race. 

The best model in this project consists of 16 convolutional layers, 3 fully connected layers for each class (age, gender, and race), and a final 8-way softmax for age class, 2-way softmax for gender class, and 5-way softmax for race class. 

To reduce overfitting, this project  
1) added max-pooling layer and batch normalization between successive convolutional layers and dropout layer before each fully connected layer,  
2) applied data augmentation, and  
3) early stopping.

## 1 Dataset

### 1.1 Overview

<img width="800" alt="image" src="https://user-images.githubusercontent.com/74934323/102843052-64e93700-43d6-11eb-86fd-89eb02053abd.png">
The dataset this project used is UTKFace dataset, which is released by UCI machine learning repository, consists of over 20,000 face images with annotations of age, gender, and ethnicity. As shown in the picture above, although the images are properly cropped and only contains the face region, there are variations in pose, facial expression, illumination. etc. among images, and thus this project thinks data augmentation is needed.     

  
More information of this dataset please check [this website](http://aicip.eecs.utk.edu/wiki/UTKFace).

 
### 1.2 Distribution of Gender, Age, and Race in the dataset 
 
#### 1.2.1 Gender
 
<img width="450" alt="image" src="https://user-images.githubusercontent.com/74934323/102845400-7e40b200-43db-11eb-8c0b-37c2de6c7901.png">
 
  
#### 1.2.2 Age
  
<img width="450" alt="image" src="https://user-images.githubusercontent.com/74934323/102845312-4afe2300-43db-11eb-9c55-d45d8c18d6f8.png">
 
 
#### 1.2.2 Race
   
<img width="450" alt="image" src="https://user-images.githubusercontent.com/74934323/102845506-b34d0480-43db-11eb-8734-8618b7c4ddc8.png">


## 2 Strategies Applied for Reducing Overfitting


### 2.1 Max-pooling

When the training dataset is not large enough to contain all the features in the whole dataset, overfitting happens. By adding max-pooling layer, the size of spatial size and the number of parameters will be reduced (et. only a subset of features which has the max value will be selected), as a result the model is less likely to learn false patterns.


### 2.2 Batch Normalization

Regularization introduces additional information to the model and thus reduce overfitting problem. One of the problems encountered often when training Deep Neural Networks is internal covariate shift, which is caused by the different distribution of each layer’s inputs. Luckily, batch normalization can avoid the problem by reducing the amount of hidden unit values shift around. Also, each layer of the network can learn more independently from other layers 


### 2.3 Dropout 

One of the major challenges in training Deep Neural Network is deciding when to stop the training. With too short time of training, underfitting occurs, while with too long time of training, overfitting occurs. This project will also apply early stopping to reduce overfitting – stop training when performance on the validation dataset starts to degrade


### 2.4 Data Augmentation

Data Augmentation reduces overfitting by enlarging the features in the training set and avoid the training model learning false patterns. Data Augmentation has been used widely and show effective results in image classification 

Below an example after applying data augmentation of one of the images in the dataset:

<img width="450" alt="image" src="https://user-images.githubusercontent.com/74934323/102846161-46d30500-43dd-11eb-86c5-39cc9264437d.png">



## 3 Model Architecture

### 3.3.1 Model 1

<img width="800" alt="image" src="https://user-images.githubusercontent.com/74934323/102846623-78000500-43de-11eb-834a-3ed14a60d0bf.png">

Model 1 has 5 convolutional layers, and 3 fully connectect layers for each output.


### 3.3.2 Model 2

<img width="800" alt="image" src="https://user-images.githubusercontent.com/74934323/102846657-8e0dc580-43de-11eb-8716-03df04544ddb.png">

Model2 has 10 convolutional layers, and 3 fully connectect layers for each output.


### 3.3.3 Model 3

<img width="800" alt="image" src="https://user-images.githubusercontent.com/74934323/102846677-9cf47800-43de-11eb-9035-904cad0549fa.png">


Model3 has 16 convolutional layers with residual learning, and 3 fully connectect layers for each output.



## 4 Result

Model 3 has the best performance, after adding simple convolutional layers and apply residual learning, the accuracies for image classification have been improved on age, gender, and race.  With model 3, the accuracy for classifying age, gender, and race are 59.49%, 91.22%, and 81.23%.

Below is the results predicted by model 3 of 16 randomly selected images : 

<img width="500" alt="image" src="https://user-images.githubusercontent.com/74934323/102849124-746f7c80-43e4-11eb-8105-7bf233eadde8.png">



## References
UTK Face Dataset: http://aicip.eecs.utk.edu/wiki/UTKFace

Keras Multi-output documentation: https://keras.io/getting-started/functional-api-guide/

Krizhevsky, A., Sutskever, I., and Hinton, G.E. Imagenet classification with deep convolutional neural networks. In NIPS, 2012. 

Karen Simonyan & Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. In ICLR, 2015

K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556,2014.	 

Wang, J., Perez, L.: The Effectiveness of Data Augmentation in Image Classification using Deep Learning. http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf

Zhang, C., Vinyals, O., Munos, R., et al.: A Study on Overfitting in Deep Reinforcement Learning. https://arxiv.org/pdf/1804.06893.pdf

Ioffe, S. & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.. CoRR, abs/1502.03167. 

Hernandez-Garcia, A., Konig, P.: Do deep nets really need weight decay and
dropout? https://arxiv.org/pdf/1802.07042.pdf 


Prechelt, L. (1996). Early Stopping-But When?. In G. B. Orr & K.-R. Müller (ed.), Neural Networks: Tricks of the Trade , Vol. 1524 (pp. 55-69) . Springer . ISBN: 3-540-65311-2.

S. C. Wong, A. Gatt, V. Stamatescu, and M. D. McDonnell.Understanding data augmentation for classification: when to warp? CoRR, abs/1609.08764, 2016.
