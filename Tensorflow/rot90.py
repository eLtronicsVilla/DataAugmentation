# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:27:49 2020

@author: eLtronicsvilla
"""

# import the module
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import PIL.Image
import math
import random
import numpy as np

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
    


#degree = random.random()*360
#degree  = random.randint(1,4)*90
mul = tf.random.uniform([],-1,2,dtype=tf.int32)
mul = tf.dtypes.cast(mul, dtype=tf.float32)
#degree = 90.0*float(mul)
degree = tf.math.scalar_mul(90.0,mul)
print(degree)
#aug = tfa.image.rotate(image, degree * math.pi / 180, interpolation='BILINEAR')
aug = tfa.image.rotate(image, degree * np.pi / 180.0, interpolation='BILINEAR')



visualize(image,aug)