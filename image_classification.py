#!/usr/bin/env python
# coding: utf-8

# # Fianl Project - Hongyu Wang

# ## 1 Data Exploration

# ### 1.1 Load Data

# In[1]:


import numpy as np
import pandas as pd
import os

wd = os.getcwd()
path = os.path.join(wd, "UTKFace")
files = os.listdir(path)
size = len(files)
print("Total sample size is", size)


# In[231]:


# create three lists to store information of age, gender, race, and image respectively
import cv2
ages = []
genders = []
races = []
images = []

for file in files:
    try:
        age = int(file.split('_')[0]) # string to int
        gender = int(file.split('_')[1]) # string to int
        race = int(file.split('_')[2])

        img = cv2.imread(path+'/'+file) # Using 0 to read image in grayscale mode
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dim = (96, 96)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA) # resize the original image to 60 * 60 * 3

        ages.append(age)
        genders.append(gender)
        races.append(race)
        images.append(img)
    except Exception as ex:
        continue


# ### 1.2 Data Visualization

# #### 1.2.1 Gender 

# In[174]:


# visualize gender distribution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.DataFrame({"Gender": genders, "Age": ages})

fig, ax = plt.subplots()
sns.set_context("talk")
sns.set_palette(["red", "blue"])
sns.countplot(x = "Gender", data = df, order = [0,1])
ax.set_xticklabels(["Female", "Male"])
ax.set_title("Distribution of Gender")
plt.show()
fig.savefig('gender_distribution.png', dpi = 300, bbox_inches='tight')


# We can see that the distribution of gender is relatively balanced, with a little bit more of female.

# #### 1.2.2 Age

# In[175]:


# visualize age distribution
df['Counts'] = df.groupby(['Age'])['Age'].transform('count')

fig, ax = plt.subplots()
sns.set_context("talk")
sns.lineplot(x = "Age",y = "Counts", data = df, color = "blue")
ax.set_title("Distribution of Age")
plt.show()
fig.savefig('age_distribution.png', dpi = 300, bbox_inches='tight')


# We can see that most of the samples are below age 50, and thus we may get skewed training dataset. I will try to solve this problem during data preprocessing.

# #### 1.2.3 Race

# In[176]:


temp = pd.DataFrame({"races":races})
g = sns.countplot(x = "races", data = temp)
g.set_title("Distribution of Race")
g.set_xticklabels(["white", "black", "asian", "indian", "others"])
plt.show()
g.figure.savefig('race_distribution.png', dpi = 300, bbox_inches='tight')


# #### 1.2.4 Image

# In[12]:


def visualize_data():
    images_to_show = 36
    per_row = 12
    fig = plt.figure(figsize=(20,5))
    for i in range(images_to_show):
        pos = (i // per_row, i % per_row)
        ax = plt.subplot2grid((int(images_to_show / per_row), per_row),
                              pos, xticks=[], yticks=[])
        ax.imshow(images[i])
    plt.show()
    fig.savefig('face_demo.png', dpi = 300, bbox_inches='tight')
    
visualize_data()


# ## 2 Data Preprocessing

# ### 2.1 One Hot Encode

# #### 2.1.1 Age

# I divide ages into 8 age groups:
# - class1: 0 <= age < 6
# - class2: 6 <= age < 12
# - class3: 12 <= age < 25
# - class4: 25 <= age < 35
# - class5: 35 <= age < 45
# - class6: 45 <= age < 60
# - class7: 60 <= age < 80
# - class8: age >= 80

# In[232]:


def age_group(age):
    if age >=0 and age < 6:
        return "class1"
    elif age < 12:
        return "class2"
    elif age < 25:
        return "class3"
    elif age < 35:
        return "class4"
    elif age < 45:
        return "class5"
    elif age < 60:
        return "class6"
    elif age < 80:
        return "class7"
    else:
        return "class8"

for i in range(len(ages)):
    temp = ages[i]
    ages[i] = age_group(temp)
    
# The number of age categories
n_age_categories = 8
# The unique values of categories in the data
age_categories = np.array(["class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8"])
# initialize ohe_labels as all zeros
age_ohe_labels = np.zeros((len(ages), n_age_categories))
# loope over the labels
for i in range(len(ages)):
    age_ohe_labels[i] = np.where(age_categories == ages[i], 1, 0)


# #### 2.1.2 Gender

# According to the dataset description:
# - 0: male
# - 1: female

# In[233]:


def gender_group(gender):
    if gender == 0:
        return "male"
    else:
        return "female"
    
for i in range(len(genders)):
    temp = genders[i]
    genders[i] = gender_group(temp)

n_gender_categories = 2
gender_categories = np.array(["male", "female"])
gender_ohe_labels = np.zeros((len(genders), n_gender_categories))
for i in range(len(genders)):
    gender_ohe_labels[i] = np.where(gender_categories == genders[i], 1, 0)


# #### 2.1.3 Race

# According to the decription of the dataset:
# - 0: white
# - 1: black
# - 2: asian
# - 3: indian
# - 4: others

# In[234]:


def race_group(race):
    if race == 0:
        return "white"
    elif race == 1:
        return "black"
    elif race == 2:
        return "asian"
    elif race == 3:
        return "indian"
    else:
        return "others"
    
for i in range(len(races)):
    temp = races[i]
    races[i] = race_group(temp)
    
n_race_categories = 5
race_categories = np.array(["white", "black", "asian", "indian", "others"])
race_ohe_labels = np.zeros((len(races), n_race_categories))
for i in range(len(races)):
    race_ohe_labels[i] = np.where(race_categories == races[i], 1, 0)


# ### 2.2 Prepare Training, Validation, and Test Sets

# In[32]:




img_features = np.array(images)
img_features = img_features / 255 # normalize images


n1 = int(len(img_features) * 0.75)
n2 = int((len(img_features) - n1) / 4) + n1
# training set
x_train, y_train = img_features[:n1], [age_ohe_labels[:n1], gender_ohe_labels[:n1], race_ohe_labels[:n1]]
# test set
x_test, y_test = img_features[n1:n2], [age_ohe_labels[n1:n2], gender_ohe_labels[n1:n2], race_ohe_labels[n1:n2]]
# valid set
x_valid, y_valid = img_features[n2:], [age_ohe_labels[n2:], gender_ohe_labels[n2:], race_ohe_labels[n2:]]


# ### 2.3 Data Augmentation

# #### 2.3.1 Data Augmentation Generator

# In[33]:


from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(
            width_shift_range=.15,
            height_shift_range=.15,
            rotation_range=15,
            horizontal_flip=True,
            zoom_range=[0.5,1.0])


# #### 2.3.2 Image Data Augmentation Result Example

# In[143]:


from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def plot_augmentation(image):
    data = img_to_array(image)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    image_gen = ImageDataGenerator(
            width_shift_range=.15,
            height_shift_range=.15,
            rotation_range=15,
            horizontal_flip=True,
            zoom_range=[0.5,1.0])
    # prepare iterator
    it = image_gen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        plt.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        plt.imshow(image)
    # show the figure
    plt.show()
    plt.savefig('augmentation_demo.png', dpi = 300, bbox_inches='tight')
    
    
plot_augmentation(images[6])


# #### 2.3.3 Data Augmentation Generator for Model Training

# In[53]:


batch_size = 32
def generator(x_train, y_train):
    genX1 = image_gen.flow(x_train,y_train[0], batch_size=batch_size,seed=1)
    genX2 = image_gen.flow(x_train,y_train[1], batch_size=batch_size,seed=1)
    genX3 = image_gen.flow(x_train,y_train[2], batch_size=batch_size,seed=1)
    
    while True:
            X1 = genX1.next()
            X2 = genX2.next()
            X3 = genX3.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            X = X1[0]/255.0
            yield X, [X1[1], X2[1], X3[1]]
            
images_temp = np.array(images)
x_train_temp, y_train_temp = images_temp[:n1], [age_ohe_labels[:n1], gender_ohe_labels[:n1], race_ohe_labels[:n1]]
# create generator
gen_flow = generator(x_train_temp, [y_train_temp[0], y_train_temp[1], y_train_temp[2]])


# ## 3 Model Building & Training

# ### 3.1 Hyperparameter Tuning

# #### 3.1.1 Optimizer

# In[180]:


def get_alexnet_model(optimizer):
    np.random.seed(1000)
    inputs = Input(shape = (96, 96, 3))

    # 1st Convolutional Layer
    conv1 = Conv2D(filters=96, kernel_size=(2,2), strides=(2,2), padding="valid", activation = "relu")(inputs)
    # Max Pooling
    maxp1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv1)
    maxp1 = BatchNormalization() (maxp1)

    # 2nd Convolutional Layer
    conv2 = Conv2D(filters=256, kernel_size=(2,2), strides=(1,1), padding="same", activation = "relu") (maxp1)
    # Max Pooling
    maxp2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv2)
    maxp2 = BatchNormalization() (maxp2)

    # 3rd Convolutional Layer
    conv3 = Conv2D(filters=384, kernel_size=(2,2), strides=(1,1), padding="valid", activation = "relu")(maxp2)

    # 4th Convolutional Layer
    conv4 = Conv2D(filters=384, kernel_size=(2,2), strides=(1,1), padding="valid", activation = "relu")(conv3)

    # 5th Convolutional Layer
    conv5 = Conv2D(filters=256, kernel_size=(2,2), strides=(1,1), padding="valid", activation = "relu")(conv4)
    # Max Pooling
    maxp3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv5)

    # Passing it to a Fully Connected layer
    flatten = Flatten()(maxp3)


    # 1st Fully Connected Layer
    dense_1 = Dense(1024, activation = 'relu') (flatten)
    dense_1 = BatchNormalization() (dense_1)
    dense_2 = Dense(1024, activation = 'relu') (flatten)
    dense_2 = BatchNormalization() (dense_2)
    dense_3 = Dense(1024, activation = 'relu') (flatten)
    dense_3 = BatchNormalization() (dense_3)

    drop_1 = Dropout(0.4) (dense_1)
    drop_2 = Dropout(0.4) (dense_2)
    drop_3 = Dropout(0.4) (dense_3)

    # 2nd Fully Connected Layer
    dense_1 = Dense(1024, activation = 'relu') (flatten)
    dense_1 = BatchNormalization() (dense_1)
    dense_2 = Dense(1024, activation = 'relu') (flatten)
    dense_2 = BatchNormalization() (dense_2)
    dense_3 = Dense(1024, activation = 'relu') (flatten)
    dense_3 = BatchNormalization() (dense_3)

    drop_1 = Dropout(0.4) (dense_1)
    drop_2 = Dropout(0.4) (dense_2)
    drop_3 = Dropout(0.4) (dense_3)

    # 3rd Fully Connected Layer
    dense_1 = Dense(1024, activation = 'relu') (flatten)
    dense_1 = BatchNormalization() (dense_1)
    dense_2 = Dense(1024, activation = 'relu') (flatten)
    dense_2 = BatchNormalization() (dense_2)
    dense_3 = Dense(1024, activation = 'relu') (flatten)
    dense_3 = BatchNormalization() (dense_3)

    drop_1 = Dropout(0.4) (dense_1)
    drop_2 = Dropout(0.4) (dense_2)
    drop_3 = Dropout(0.4) (dense_3)

    output_1 = Dense(n_age_categories, activation = "softmax", name = "age_out") (drop_1)
    output_2 = Dense(n_gender_categories, activation = "sigmoid", name = "gender_out") (drop_2)
    output_3 = Dense(n_race_categories, activation = "softmax", name = "race_out") (drop_3)

    model = Model(inputs = [inputs], outputs = [output_1, output_2, output_3])
    
    model.compile(loss = {"gender_out":"binary_crossentropy", 
                                       "age_out":"categorical_crossentropy",
                                        "race_out":"categorical_crossentropy"}, 
                               optimizer = optimizer, 
                               metrics = ["accuracy"])
    
    return model



# In[189]:


from keras.optimizers import Adam, SGD, RMSprop, Adamax

# optimizers
adam = Adam()
sgd = SGD()
rmsprop = RMSprop()
adamax = Adamax()

optimizers = [sgd, rmsprop, adamax]

optimizer_results = {}
for optimizer in optimizers:
    model = get_alexnet_model(optimizer = optimizer)
    history = model.fit_generator(gen_flow,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=50,
                        validation_data=(x_valid, y_valid))
    optimizer_results[str(optimizer)] = history


# In[190]:


optimizer_results.items()


# In[219]:


import pandas as pd

# extract val_loss history of each optimizer
val_loss_per_optimizer = {k: v.history["val_loss"] for k, v in optimizer_results.items()}

# turn the dictionary into a pandas dataframe
val_loss_curves = pd.DataFrame(val_loss_per_optimizer)
val_loss_curves.columns = ["Adam", "SGD", "RMSprop", "Adamax"]

# plot the result
ax = val_loss_curves.plot(title = "Loss per Optimizer", color = ["c", "b", "g", "r"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")


# #### 3.1.1 Number of Filter, Dropout Rate, Learning Rate, and Decay Rate

# In[241]:


from kerastuner import HyperModel


class MyHyperModel(HyperModel):
    def build(self, hp):
        inputs = Input(shape = (96, 96, 3))

        # 1st Convolutional Layer
        conv1 = Conv2D(filters=hp.Choice(
                    'num_filter_conv1',
                    values=[64, 96, 128],
                    default=96,
                ), kernel_size=(2,2), strides=(2,2), padding="valid", activation = "relu")(inputs)
        
        # Max Pooling
        maxp1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv1)
        maxp1 = BatchNormalization() (maxp1)

        # 2nd Convolutional Layer
        conv2 = Conv2D(filters=hp.Choice(
                    'num_filter_conv2',
                    values=[96, 128, 256],
                    default=256,
                ), kernel_size=(2,2), strides=(1,1), padding="same", activation = "relu") (maxp1)
        # Max Pooling
        maxp2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv2)
        maxp2 = BatchNormalization() (maxp2)

        # 3rd Convolutional Layer
        conv3 = Conv2D(filters=hp.Choice(
                    'num_filter_conv3',
                    values=[96, 256, 384],
                    default=384,
                ), kernel_size=(2,2), strides=(1,1), padding="valid", activation = "relu")(maxp2)

        # 4th Convolutional Layer
        conv4 = Conv2D(filters=hp.Choice(
                    'num_filter_conv4',
                    values=[96, 256, 384],
                    default=384,
                ), kernel_size=(2,2), strides=(1,1), padding="valid", activation = "relu")(conv3)

        # 5th Convolutional Layer
        conv5 = Conv2D(filters=hp.Choice(
                    'num_filter_conv5',
                    values=[96, 128, 256],
                    default=256
                ), kernel_size=(2,2), strides=(1,1), padding="valid", activation = "relu")(conv4)
        # Max Pooling
        maxp3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv5)

        # Passing it to a Fully Connected layer
        flatten = Flatten()(maxp3)


        # 1st Fully Connected Layer
        dense_1 = Dense(1024, activation = 'relu') (flatten)
        dense_1 = BatchNormalization() (dense_1)
        dense_2 = Dense(1024, activation = 'relu') (flatten)
        dense_2 = BatchNormalization() (dense_2)
        dense_3 = Dense(1024, activation = 'relu') (flatten)
        dense_3 = BatchNormalization() (dense_3)

        drop_1 = Dropout(rate=hp.Float(
                'dropout_1_1',
                min_value=0.2,
                max_value=0.5,
                default=0.4,
                step=0.05)) (dense_1)
        drop_2 = Dropout(rate=hp.Float(
                'dropout_1_2',
                min_value=0.2,
                max_value=0.5,
                default=0.4,
                step=0.05)) (dense_2)
        drop_3 = Dropout(rate=hp.Float(
                'dropout_1_3',
                min_value=0.2,
                max_value=0.5,
                default=0.4,
                step=0.05)) (dense_3)

        # 2nd Fully Connected Layer
        dense_1 = Dense(1024, activation = 'relu') (flatten)
        dense_1 = BatchNormalization() (dense_1)
        dense_2 = Dense(1024, activation = 'relu') (flatten)
        dense_2 = BatchNormalization() (dense_2)
        dense_3 = Dense(1024, activation = 'relu') (flatten)
        dense_3 = BatchNormalization() (dense_3)

        drop_1 = Dropout(rate=hp.Float(
                'dropout_2_1',
                min_value=0.2,
                max_value=0.5,
                default=0.4,
                step=0.05)) (dense_1)
        drop_2 = Dropout(rate=hp.Float(
                'dropout_2_2',
                min_value=0.2,
                max_value=0.5,
                default=0.4,
                step=0.05)) (dense_2)
        drop_3 = Dropout(rate=hp.Float(
                'dropout_2_3',
                min_value=0.2,
                max_value=0.5,
                default=0.4,
                step=0.05)) (dense_3)

        # 3rd Fully Connected Layer
        dense_1 = Dense(1024, activation = 'relu') (flatten)
        dense_1 = BatchNormalization() (dense_1)
        dense_2 = Dense(1024, activation = 'relu') (flatten)
        dense_2 = BatchNormalization() (dense_2)
        dense_3 = Dense(1024, activation = 'relu') (flatten)
        dense_3 = BatchNormalization() (dense_3)

        drop_1 = Dropout(rate=hp.Float(
                'dropout_3_1',
                min_value=0.2,
                max_value=0.5,
                default=0.4,
                step=0.05)) (dense_1)
        drop_2 = Dropout(rate=hp.Float(
                'dropout_3_2',
                min_value=0.2,
                max_value=0.5,
                default=0.4,
                step=0.05)) (dense_2)
        drop_3 = Dropout(rate=hp.Float(
                'dropout_3_3',
                min_value=0.2,
                max_value=0.5,
                default=0.4,
                step=0.05)) (dense_3)

        output_1 = Dense(n_age_categories, activation = "softmax", name = "age_out") (drop_1)
        output_2 = Dense(n_gender_categories, activation = "sigmoid", name = "gender_out") (drop_2)
        output_3 = Dense(n_race_categories, activation = "softmax", name = "race_out") (drop_3)

        model = Model(inputs = [inputs], outputs = [output_1, output_2, output_3])

        # optimizer
        adamax = Adamax(lr = hp.Float(
                    'learning_rate',
                    min_value=1e-6,
                    max_value=1e-3,
                    sampling='LOG',
                    default=1e-4
                ), decay = hp.Float(
                    'decay',
                    min_value=1e-7,
                    max_value=1e-4,
                    sampling='LOG',
                    default=1e-6
                ))
        model.compile(loss = {"gender_out":"binary_crossentropy", 
                                               "age_out":"categorical_crossentropy",
                                                "race_out":"categorical_crossentropy"}, 
                                       optimizer = adamax, 
                                       metrics = ["accuracy"])
                         
        return model


# In[237]:





# In[401]:


from kerastuner.tuners import Hyperband

hypermodel = MyHyperModel()

max_epochs = 40
executions_per_trial = 2

tuner = Hyperband(
    hypermodel,
    max_epochs=max_epochs,
    
    ,
    seed=767,
    executions_per_trial=executions_per_trial,
    directory='hyperband',
    project_name='767 project tuning'
)

tuner.search_space_summary()


# In[ ]:


N_EPOCH_SEARCH = 50

tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1)


# In[233]:


# tuner.search(gen_flow,steps_per_epoch=x_train.shape[0] // batch_size,epochs=N_EPOCH_SEARCH,validation_data=(x_valid, y_valid))


# ### 3.2 Modle Building

# #### 3.2.1 Model 1

# Model 1 has 5 convolutional layers, and 3 fully connectect layers for each output.

# In[6]:


from IPython.display import Image
Image(filename = "model1.png")


# In[57]:


import keras
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model
from keras.optimizers import Adamax
import numpy as np
np.random.seed(1000)
inputs = Input(shape = (96, 96, 3))

# 1st Convolutional Layer
conv1 = Conv2D(filters=96, kernel_size=(2,2), strides=(2,2), padding="valid", activation = "relu")(inputs)
# Max Pooling
maxp1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv1)
maxp1 = BatchNormalization() (maxp1)

# 2nd Convolutional Layer
conv2 = Conv2D(filters=128, kernel_size=(2,2), strides=(1,1), padding="same", activation = "relu") (maxp1)
# Max Pooling
maxp2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv2)
maxp2 = BatchNormalization() (maxp2)

# 3rd Convolutional Layer
conv3 = Conv2D(filters=96, kernel_size=(2,2), strides=(1,1), padding="valid", activation = "relu")(maxp2)

# 4th Convolutional Layer
conv4 = Conv2D(filters=256, kernel_size=(2,2), strides=(1,1), padding="valid", activation = "relu")(conv3)

# 5th Convolutional Layer
conv5 = Conv2D(filters=128, kernel_size=(2,2), strides=(1,1), padding="valid", activation = "relu")(conv4)
# Max Pooling
maxp3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv5)

# Passing it to a Fully Connected layer
flatten = Flatten()(maxp3)


# 1st Fully Connected Layer
dense_1 = Dense(1024, activation = 'relu') (flatten)
dense_1 = BatchNormalization() (dense_1)
dense_2 = Dense(1024, activation = 'relu') (flatten)
dense_2 = BatchNormalization() (dense_2)
dense_3 = Dense(1024, activation = 'relu') (flatten)
dense_3 = BatchNormalization() (dense_3)

drop_1 = Dropout(0.5) (dense_1)
drop_2 = Dropout(0.2) (dense_2)
drop_3 = Dropout(0.35) (dense_3)

# 2nd Fully Connected Layer
dense_1 = Dense(1024, activation = 'relu') (flatten)
dense_1 = BatchNormalization() (dense_1)
dense_2 = Dense(1024, activation = 'relu') (flatten)
dense_2 = BatchNormalization() (dense_2)
dense_3 = Dense(1024, activation = 'relu') (flatten)
dense_3 = BatchNormalization() (dense_3)

drop_1 = Dropout(0.25) (dense_1)
drop_2 = Dropout(0.3) (dense_2)
drop_3 = Dropout(0.25) (dense_3)

# 3rd Fully Connected Layer
dense_1 = Dense(1024, activation = 'relu') (flatten)
dense_1 = BatchNormalization() (dense_1)
dense_2 = Dense(1024, activation = 'relu') (flatten)
dense_2 = BatchNormalization() (dense_2)
dense_3 = Dense(1024, activation = 'relu') (flatten)
dense_3 = BatchNormalization() (dense_3)

drop_1 = Dropout(0.35) (dense_1)
drop_2 = Dropout(0.4) (dense_2)
drop_3 = Dropout(0.25) (dense_3)

output_1 = Dense(n_age_categories, activation = "softmax", name = "age_out") (drop_1)
output_2 = Dense(n_gender_categories, activation = "sigmoid", name = "gender_out") (drop_2)
output_3 = Dense(n_race_categories, activation = "softmax", name = "race_out") (drop_3)

model = Model(inputs = [inputs], outputs = [output_1, output_2, output_3])

# optimizer
adam = Adamax(lr = 0.00088483, decay = 1.096e-07)
model.compile(loss = {"gender_out":"binary_crossentropy", 
                                       "age_out":"categorical_crossentropy",
                                        "race_out":"categorical_crossentropy"}, 
                               optimizer = adam, 
                               metrics = ["accuracy"])

model.summary()


# In[59]:


from keras.callbacks import ModelCheckpoint, EarlyStopping


checkpointer = ModelCheckpoint("best_model1.h5", monitor='val_loss',verbose=1,save_best_only=True)
early_stop= EarlyStopping(patience=50, monitor='val_loss',restore_best_weights=True),
callback_list=[checkpointer,early_stop]

history1 = model.fit_generator(gen_flow,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=100,
                        validation_data=(x_valid, y_valid),
                        callbacks = callback_list)


# In[ ]:





# #### 3.2.2 Model 2 (Adding more convolutional layers)

# Model2 has 10 convolutional layers, and 3 fully connectect layers for each output.

# In[7]:


Image(filename = "model2.png")


# In[177]:


from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
np.random.seed(1000)

inputs = Input(shape = (96, 96, 3))


conv1 = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation = "relu")(inputs)
conv2 = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation = "relu")(conv1)
# Max Pooling
maxp1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv2)
maxp1 = BatchNormalization()(maxp1)

conv3 = Conv2D(filters=128, kernel_size=(3,3),  padding="same", activation = "relu")(maxp1)
conv4 = Conv2D(filters=128, kernel_size=(3,3),  padding="same", activation = "relu")(conv3)
# Max Pooling
maxp2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv4)
maxp2 = BatchNormalization()(maxp2)

conv5 = Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation = "relu") (maxp2)
conv6 = Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation = "relu") (conv5)
conv7 = Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation = "relu") (conv6)
# Max Pooling
maxp3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv7)
maxp3 = BatchNormalization()(maxp3)

conv8 = Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation = "relu") (maxp3)
conv9 = Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation = "relu") (conv8)
conv10 = Conv2D(filters=256, kernel_size=(3,3), padding="valid", activation = "relu") (conv9)
# Max Pooling
maxp4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(conv10)
maxp4 = BatchNormalization()(maxp4)


# Passing it to a Fully Connected layer
flatten = Flatten()(maxp4)

# 1st Fully Connected Layer
dense_1 = Dense(1024, activation = 'relu') (flatten)
dense_1 = BatchNormalization() (dense_1)
dense_2 = Dense(1024, activation = 'relu') (flatten)
dense_2 = BatchNormalization() (dense_2)
dense_3 = Dense(1024, activation = 'relu') (flatten)
dense_3 = BatchNormalization() (dense_3)

drop_1 = Dropout(0.5) (dense_1)
drop_2 = Dropout(0.2) (dense_2)
drop_3 = Dropout(0.35) (dense_3)

# 2nd Fully Connected Layer
dense_1 = Dense(1024, activation = 'relu') (flatten)
dense_1 = BatchNormalization() (dense_1)
dense_2 = Dense(1024, activation = 'relu') (flatten)
dense_2 = BatchNormalization() (dense_2)
dense_3 = Dense(1024, activation = 'relu') (flatten)
dense_3 = BatchNormalization() (dense_3)

drop_1 = Dropout(0.25) (dense_1)
drop_2 = Dropout(0.3) (dense_2)
drop_3 = Dropout(0.25) (dense_3)

# 3rd Fully Connected Layer
dense_1 = Dense(1024, activation = 'relu') (flatten)
dense_1 = BatchNormalization() (dense_1)
dense_2 = Dense(1024, activation = 'relu') (flatten)
dense_2 = BatchNormalization() (dense_2)
dense_3 = Dense(1024, activation = 'relu') (flatten)
dense_3 = BatchNormalization() (dense_3)

drop_1 = Dropout(0.35) (dense_1)
drop_2 = Dropout(0.4) (dense_2)
drop_3 = Dropout(0.25) (dense_3)

output_1 = Dense(n_age_categories, activation = "softmax", name = "age_out") (drop_1)
output_2 = Dense(n_gender_categories, activation = "sigmoid", name = "gender_out") (drop_2)
output_3 = Dense(n_race_categories, activation = "softmax", name = "race_out") (drop_3)

model2 = Model(inputs = [inputs], outputs = [output_1, output_2, output_3])

# optimizer
adam = Adamax(lr = 0.00088483, decay = 1.096e-07)
model2.compile(loss = {"gender_out":"binary_crossentropy", 
                                       "age_out":"categorical_crossentropy",
                                        "race_out":"categorical_crossentropy"}, 
                               optimizer = adam, 
                               metrics = ["accuracy"])

model2.summary()


# In[178]:


from keras.callbacks import ModelCheckpoint, EarlyStopping


checkpointer = ModelCheckpoint("best_model3.h5", monitor='val_loss',verbose=1,save_best_only=True)
early_stop= EarlyStopping(patience=50, monitor='val_loss',restore_best_weights=True),
callback_list=[checkpointer,early_stop]

history2 = model2.fit_generator(gen_flow,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=100,
                        validation_data=(x_valid, y_valid),
                        callbacks = callback_list)


# In[ ]:





# #### 3.2.3 Model 3 (Adding more convolutional layers with residual learning)

# Model2 has 16 convolutional layers with residual learning, and 3 fully connectect layers for each output.

# In[4]:


Image(filename = "model3.png")


# In[192]:


from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
np.random.seed(1000)
from keras.layers import Add, Dense, Activation, ZeroPadding2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.initializers import glorot_uniform

# standard block - when input sha4pe and output shape are the same
def identity_block(x, filters, stage, block):
    conv_layer_name = "convstage%dblock%d"%(stage, block)
    bn_layer_name = "bnstate%dblock%d"%(stage, block)
    
    filter1, filter2 = filters
    
    x_shortcut = x
    
    # fist component
    x = Conv2D(filters = filter1, kernel_size = (1,1),  kernel_initializer = glorot_uniform(seed=0), name = conv_layer_name + "-A")(x)
    x = BatchNormalization(axis = 3, name = bn_layer_name + "A")(x)
    x = Activation("relu")(x)
    
    # second component
    x = Conv2D(filters = filter1, kernel_size = (3,3), padding = "same", kernel_initializer = glorot_uniform(seed=0), name = conv_layer_name + "-B")(x)
    x = BatchNormalization(axis = 3, name = bn_layer_name + "B")(x)
    x = Activation("relu")(x)
    
    # third component
    x = Conv2D(filters = filter2, kernel_size = (1,1),  kernel_initializer = glorot_uniform(seed=0), name = conv_layer_name + "-C")(x)
    x = BatchNormalization(axis = 3, name = bn_layer_name + "C")(x)
    
    x = Add()([x, x_shortcut])
    x = Activation("relu")(x)
    
    # Max Pooling
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(x)
    x = BatchNormalization()(x)
    
    return x

# this block is used when input shape and output shape are different
def convolutional_block(x, filters, stage, block):
    conv_layer_name = "convstage%dblock%d"%(stage, block)
    bn_layer_name = "bnstate%dblock%d"%(stage, block)
    
    filter1, filter2 = filters
    
    x_shortcut = x
    
    # fist component
    x = Conv2D(filters = filter1, kernel_size = (1,1), kernel_initializer = glorot_uniform(seed=0), name = conv_layer_name + "-A")(x)
    x = BatchNormalization(axis = 3, name = bn_layer_name + "A")(x)
    x = Activation("relu")(x)
    
    # second component
    x = Conv2D(filters = filter1, kernel_size = (3,3), padding = "same", kernel_initializer = glorot_uniform(seed=0), name = conv_layer_name + "-B")(x)
    x = BatchNormalization(axis = 3, name = bn_layer_name + "B")(x)
    
    # third component
    x = Conv2D(filters = filter2, kernel_size = (3,3), padding = "same", kernel_initializer = glorot_uniform(seed=0), name = conv_layer_name + "-C")(x)
    x = BatchNormalization(axis = 3, name = bn_layer_name + "C")(x)
    
    # add a convolutional layer in the shortcut path
    x_shortcut = Conv2D(filters = filter2, kernel_size = (1,1), strides = (2,2), kernel_initializer = glorot_uniform(seed=0), name = conv_layer_name + "shortcut_path" + str(stage))(x_shortcut)
    x_shortcut = BatchNormalization(axis = 3, name = bn_layer_name + "short_cutpath" + str(stage))(x)
    
    x = Add()([x, x_shortcut])
    x = Activation("relu")(x)
    
    # Max Pooling
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(x)
    x = BatchNormalization()(x)
    
    return x
    
    
# implementation
input_shape = (96, 96, 3)
inputs = Input(input_shape)

# zero-padding
x = ZeroPadding2D((3,3))(inputs)

# stage 1
x = Conv2D(64, kernel_size = (3,3), strides = (2,2), name = "convstage1")(x)
x = BatchNormalization(axis = 3, name = "bnstage1")(x)
x = Activation("relu")(x)
x = MaxPooling2D((3,3), strides = (1,1))(x)

# stage 2
x = convolutional_block(x, [64, 128], 2, 1)
x = identity_block(x, [128, 128], 2, 2)
x = identity_block(x, [128, 128], 2, 3)

# stage 3
x = convolutional_block(x, [128, 256], 3, 1)
x = identity_block(x, [256, 256], 3, 2)

# Passing it to a Fully Connected layer
flatten = Flatten()(x)

# 1st Fully Connected Layer
dense_1 = Dense(1024, activation = 'relu') (flatten)
dense_1 = BatchNormalization() (dense_1)
dense_2 = Dense(1024, activation = 'relu') (flatten)
dense_2 = BatchNormalization() (dense_2)
dense_3 = Dense(1024, activation = 'relu') (flatten)
dense_3 = BatchNormalization() (dense_3)

drop_1 = Dropout(0.5) (dense_1)
drop_2 = Dropout(0.2) (dense_2)
drop_3 = Dropout(0.35) (dense_3)

# 2nd Fully Connected Layer
dense_1 = Dense(1024, activation = 'relu') (flatten)
dense_1 = BatchNormalization() (dense_1)
dense_2 = Dense(1024, activation = 'relu') (flatten)
dense_2 = BatchNormalization() (dense_2)
dense_3 = Dense(1024, activation = 'relu') (flatten)
dense_3 = BatchNormalization() (dense_3)

drop_1 = Dropout(0.25) (dense_1)
drop_2 = Dropout(0.3) (dense_2)
drop_3 = Dropout(0.25) (dense_3)

# 3rd Fully Connected Layer
dense_1 = Dense(1024, activation = 'relu') (flatten)
dense_1 = BatchNormalization() (dense_1)
dense_2 = Dense(1024, activation = 'relu') (flatten)
dense_2 = BatchNormalization() (dense_2)
dense_3 = Dense(1024, activation = 'relu') (flatten)
dense_3 = BatchNormalization() (dense_3)

drop_1 = Dropout(0.35) (dense_1)
drop_2 = Dropout(0.4) (dense_2)
drop_3 = Dropout(0.25) (dense_3)

output_1 = Dense(n_age_categories, activation = "softmax", name = "age_out") (drop_1)
output_2 = Dense(n_gender_categories, activation = "sigmoid", name = "gender_out") (drop_2)
output_3 = Dense(n_race_categories, activation = "softmax", name = "race_out") (drop_3)

model3 = Model(inputs = [inputs], outputs = [output_1, output_2, output_3])

# optimizer
adam = Adamax(lr = 0.00088483, decay = 1.096e-07)
model3.compile(loss = {"gender_out":"binary_crossentropy", 
                                       "age_out":"categorical_crossentropy",
                                        "race_out":"categorical_crossentropy"}, 
                               optimizer = adam, 
                               metrics = ["accuracy"])

model3.summary()


# In[193]:


from keras.callbacks import ModelCheckpoint, EarlyStopping


checkpointer = ModelCheckpoint("best_model3.h5", monitor='val_loss',verbose=1,save_best_only=True)
early_stop= EarlyStopping(patience=20, monitor='val_loss',restore_best_weights=True),
callback_list=[checkpointer,early_stop]

history3 = model3.fit_generator(gen_flow,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=100,
                        validation_data=(x_valid, y_valid),
                        callbacks = callback_list)


# In[ ]:





# ## 4 Evaluation

# ##### Class for evaluation

# In[255]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.dpi']= 80
from sklearn.metrics import classification_report

sns.set_style("whitegrid")
sns.set_context("paper")

### functions for model evaluation
class Evaluation():
    def evaluation(self, name, model_name, train_accuracy, valid_accuracy):
        plt.plot(train_accuracy, label = "training accuracy")
        plt.plot(valid_accuracy, label = "validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0.3, 1)
        plt.title("%s Prediction Evaluation"%(name))
        plt.legend()
        plt.savefig(name + model_name + "evaluation.png", dpi = 300, bbox_inches='tight')
        plt.show()
        
        
    def age_evaluation_plot(self, model_name, history):
        train_accuracy = history.history["age_out_accuracy"]
        valid_accuracy = history.history["val_age_out_accuracy"]
        self.evaluation("Age", model_name, train_accuracy, valid_accuracy)

    
    def gender_evaluation_plot(self, model_name, history):
        train_accuracy = history.history["gender_out_accuracy"]
        valid_accuracy = history.history["val_gender_out_accuracy"]
        self.evaluation("Gender", model_name, train_accuracy, valid_accuracy)

        
    def race_evaluation_plot(self, model_name, history):
        train_accuracy = history.history["race_out_accuracy"]
        valid_accuracy = history.history["val_race_out_accuracy"]
        self.evaluation("Race", model_name, train_accuracy, valid_accuracy)
        
    def test_accuracy(self, model, model_name):
        loss, age_out_loss, gender_out_loss, race_out_loss, age_out_accuracy, gender_out_accuracy, race_out_accuracy =             model.evaluate(x_test, y_test, batch_size = 128, verbose = 0)
        
        age_out_accuracy = age_out_accuracy * 100
        gender_out_accuracy = gender_out_accuracy * 100
        race_out_accuracy = race_out_accuracy * 100
        total_loss = loss
        
        print("With %s: "%(model_name))
        print("The accuracy for AGE in the test set is %.2f%%"%(age_out_accuracy))
        print("The accuracy for GENDER in the test set is %.2f%%"%(gender_out_accuracy))
        print("The accuracy for RACE in the test set is %.2f%%"%(race_out_accuracy))
        print("Total loss is %.2f"%(total_loss))
     
    ### function for model report
    def get_report(self, target, pred, row_names):
        target = np.argmax(target, axis=1)
        pred = np.argmax(pred, axis=1)
        report = classification_report(target, pred, output_dict=True)
        # convert to data frame
        report_df = pd.DataFrame(report).transpose()
        # rename row names
        row_names = pd.Series(row_names)
        report_df = report_df.set_index(row_names)

        return report_df

    def report_model(self, model):
        pred_age, pred_gender, pred_race = model.predict(x_test, batch_size=32, verbose=1)
        true_age, true_gender, true_race = y_test[0], y_test[1], y_test[2]
        age_row_names = np.array(["0 <= age < 6", "6 <= age < 12", "12 <= age < 25", "25 <= age < 35", "35 <= age < 45", 
                             "45 <= age < 60", "60 <= age < 80", "age >= 80", "accuracy", "macro avg", "weighted avg"])
        gender_row_names = np.array(["male", "female", "accuracy", "macro avg", "weighted avg"])
        race_row_names = np.array(["white", "black", "asian", "indian", "others", "accuracy", "macro avg", "weighted avg"])
        # age prediction report
        age_report = self.get_report(true_age, pred_age, age_row_names)
        gender_report = self.get_report(true_gender, pred_gender, gender_row_names)
        race_report = self.get_report(true_race, pred_race, race_row_names)
        return (age_report, gender_report, race_report)


# In[256]:


# implementation
def evaluate_model(model, history, model_name):
    evaluation = Evaluation()
    evaluation.age_evaluation_plot(model_name, history)
    evaluation.gender_evaluation_plot(model_name, history)
    evaluation.race_evaluation_plot(model_name, history)
    evaluation.test_accuracy(model, model_name)
    model_age_report, model_gender_report, model_race_report = evaluation.report_model(model)
    for report in [model_age_report, model_gender_report, model_race_report]:
        display(report)


# ### 4.1 Model1 

# In[260]:


evaluate_model(model, history1, "model1")


# ### 4.2 Model2

# In[261]:


evaluate_model(model2, history2, "model2")


# ### 4.3 Model3

# In[263]:


evaluate_model(model3, history3, "model3")


# In[ ]:





# In[208]:





# ## 5 Output Visualization

# In this part I will visualize the predictions of the model with best performance (model 3).

# ### 5.1 Prepare Data

# In[222]:


pred_age, pred_gender, pred_race = model5.predict(x_test)


# In[224]:


pred_age = np.argmax(pred_age, axis=1)
pred_gender = np.argmax(pred_gender, axis=1)
pred_race = np.argmax(pred_race, axis = 1)

images_show = images[n1:n2]
true_age_list = ages[n1:n2]
true_gender_list = genders[n1:n2]
true_race_list = races[n1:n2]


def age_diaplay_class(age):
    if age == 0:
        return "0-5"
    elif age == 1:
        return "6-11"
    elif age == 2:
        return "12-24"
    elif age == 3:
        return "25-34"
    elif age == 4:
        return "35-44"
    elif age == 5:
        return "45-59"
    elif age == 6:
        return "60-79"
    else: 
        return "80+"
    
def age_diaplay_class2(age):
    if age == "class1":
        return "0-5"
    elif age == "class2":
        return "6-11"
    elif age == "class3":
        return "12-24"
    elif age == "class4":
        return "25-34"
    elif age == "class5":
        return "35-44"
    elif age == "class6":
        return "45-59"
    elif age == "class7":
        return "60-79"
    else: 
        return "80+"

for i in range(len(true_age_list)):
    temp = true_age_list[i]
    true_age_list[i] = age_diaplay_class2(temp)

pred_age_list = pred_age.tolist()
for i in range(len(pred_age_list)):
    temp = pred_age_list[i]
    pred_age_list[i] = age_diaplay_class(temp)

# for i in range(len(true_age_list)):
#     temp = true_age_list[i]
#     true_age_list[i] = age_group(temp)

pred_gender_list = pred_gender.tolist()
for i in range(len(pred_gender_list)):
    temp = pred_gender_list[i]
    pred_gender_list[i] = gender_group(temp)

pred_race_list = pred_race.tolist()
for i in range(len(pred_race_list)):
    temp = pred_race_list[i]
    pred_race_list[i] = race_group(temp)


# ### 5.2 Visualize Result

# In[241]:


import math
n = 16
random_indices = np.random.permutation(n)
n_cols = 4
n_rows = math.ceil(n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 17))
for i, img_idx in enumerate(random_indices):
    ax = axes.flat[i]
    ax.imshow(images_show[img_idx])
    
    cur_age_pred = pred_age_list[img_idx]
    cur_age_true = true_age_list[img_idx]
    
    cur_gender_pred = pred_gender_list[img_idx]
    cur_gender_true = true_gender_list[img_idx]
    
    cur_race_pred = pred_race_list[img_idx]
    cur_race_true = true_race_list[img_idx]
    
    age_threshold = 10
    if cur_gender_pred == cur_gender_true and cur_race_pred == cur_race_true and cur_age_pred == cur_age_true:
        ax.xaxis.label.set_color('green')
    elif cur_gender_pred != cur_gender_true and cur_race_pred != cur_race_true and cur_age_pred != cur_age_true:
        ax.xaxis.label.set_color('red')
    
    ax.set_xlabel('a: {}, g: {}, r: {}'.format(pred_age_list[img_idx],
                            pred_gender_list[img_idx],
                               pred_race_list[img_idx]), fontsize = 20)
    
    ax.set_title('a: {}, g: {}, r: {}'.format(true_age_list[img_idx],
                            true_gender_list[img_idx],
                               true_race_list[img_idx]), fontsize = 20)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.savefig("pred1.png", dpi = 300, bbox_inches='tight')


# In[ ]:




