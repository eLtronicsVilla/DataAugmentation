# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:56:57 2020

@author: brije
"""


# import the module
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image

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
    #plt.title('original')
    #plt.imshow(original)
    #plt.title('augumented')
    #plt.imshow(augmented)
    

aug = tf.image.random_flip_left_right(image)
aug = tf.image.flip_up_down(aug)

visualize(image,aug)