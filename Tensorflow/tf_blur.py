# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:17:08 2020

@author: brije
"""


# import the module
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import math
import random

# load dataset
image_path = tf.keras.utils.get_file("cat.jpg", "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")
PIL.Image.open(image_path)
#image = PIL.Image.open("E:\\workspace\\workspace-brijesh\\daily_dose\\TF-dose\\tf_image\\cat.png")


image_string = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image_string)


def visualize(original,augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
    
  


#filter = random.randrange(3,8,2)
#image = tf.convert_to_tensor(np.array(image))
a = [3,5,7,9]
filter= random.choice(a)
print(filter)
aug = tfa.image.mean_filter2d(image,[filter,filter], padding='REFLECT')

#jpeg_quality = 10 #must be between [0,100]
#aug = tf.image.adjust_jpeg_quality(image, jpeg_quality, name=None)



visualize(image,aug)