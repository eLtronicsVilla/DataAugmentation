# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:16:02 2020

@author: eLtronicsvilla
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
    
contrast_factor = 1.2 # delta value should be float
aug = tf.image.adjust_contrast(image, contrast_factor)
#tf.image.random_contrast(image, lower, upper, seed=None)

#aug = tf.image.adjust_brightness(image, delta)

visualize(image,aug)
