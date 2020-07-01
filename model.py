import numpy as np 
import matplotlib.pyplot as plt 
import os
from PIL import Image

# Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img

MODEL_NAME="CUSTOM"

mainDIR = os.listdir('New folder/chest-xray-pneumonia/chest_xray/chest_xray')
print(mainDIR)

train_folder= 'New folder/chest_xray/chest_xray/train/'
val_folder = 'New folder/chest_xray/chest_xray/val1/'
test_folder = 'New folder/chest_xray/chest_xray/test/'

# train 
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'

#Normal pic 
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))

sic_pic =  os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)
print(sic_load)

#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Covid 19')

# let's build the CNN model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(224,224,3)))
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(224,224,3)))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1 , activation='sigmoid'))


    print(model.summary())
    
    model.compile('adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.

training_set = train_datagen.flow_from_directory(train_folder,
                                                 target_size = (224, 224),
                                                 batch_size = 163,
                                                 class_mode = 'binary',
                                                 shuffle=True)

validation_generator = test_datagen.flow_from_directory(val_folder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False)

test_set = test_datagen.flow_from_directory(test_folder,
                                            target_size = (224,224),
                                            batch_size = 32,
                                            class_mode = 'binary',
                                            shuffle=False)

from sklearn.utils import class_weight
def get_weight(y):
    class_weight_current =  class_weight.compute_class_weight('balanced', np.unique(y), y)
    return class_weight_current

class_weight = get_weight(training_set.classes)
class_weight

cnn_model = model.fit_generator(training_set,
                         steps_per_epoch = len(training_set),
                         epochs = 25,
                         validation_data = validation_generator,
                         validation_steps = len(validation_generator),
                         class_weight=class_weight)

test_accu = model.evaluate_generator(test_set,steps=len(test_set))
print('The testing accuracy is :',test_accu[1]*100, '%')

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

import tensorflow as tf
import keras
saver=tf.train.Saver()
model.save('')

# dimensions of our images
img_width, img_height = 224, 224

# load the model we saved
model = load_model('custombinary')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

test_accu = model.evaluate_generator(test_set,steps=len(test_set))
print('The testing accuracy is :',test_accu[1]*100, '%')

model.save('Disease1.h5') 
model=load_model('Disease1.h5')

import pickle
pickle.dump(model,open('Disease1.pkl','wb'))
#accuracy 90.05%
