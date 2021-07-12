# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 09:35:50 2020

@author: eLtronicsVilla
"""


import tensorflow as tf


import PIL.Image

import matplotlib.pyplot as plt



image_path = tf.keras.utils.get_file("cat.jpg", "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")
img = PIL.Image.open(image_path)

image_string=tf.io.read_file(image_path)
image=tf.image.decode_jpeg(image_string,channels=3)




# A function to visualize original and Augmented image
def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)

    
# Flip the image either verticle or horizontal
flipped = tf.image.flip_up_down(image)
#flipped = tf.image.random_flip_up_down(image, seed=None)


print(type(image))
print(type(flipped))
visualize(image, flipped)

    
