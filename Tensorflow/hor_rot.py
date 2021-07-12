# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 12:49:12 2020

@author: eLtronicsVilla
"""


# import the module
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import PIL.Image
import math
import random

# load dataset
image_path = tf.keras.utils.get_file("cat.jpg", "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")
PIL.Image.open(image_path)

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
    

aug = tf.image.flip_left_right(image)
#aug = tf.image.rot90(aug)
degree  = 90
aug = tfa.image.rotate(aug, degree * math.pi / 180, interpolation='BILINEAR')

#aug = tfa.image.mean_filter2d(image,[5,5], padding='REFLECT')
#degree = random.random()*360
#degree  = random.randint(1,4)*90
#aug = tfa.image.rotate(image, degree * math.pi / 180, interpolation='BILINEAR')
#aug = tfa.image.rotate(image,90) 



visualize(image,aug)