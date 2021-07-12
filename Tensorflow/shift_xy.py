# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 22:11:55 2020

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

#t= tf.Variable([image.shape[1],image.shape[0]])
x_val = 90.45
y_val = 20.39
x = tf.random.uniform([],0,1.0,dtype=tf.float32,seed=random.seed(40))
x_v = tf.convert_to_tensor(x_val,dtype=tf.float32)
#mul_x = float(x*x_v)
#mul_y = x*y_val
#print(mul_x)
#t = tf.constant([mul_x,0])
t = tf.constant([0,float(x*x_v)])
print(tf.shape(t))
aug = tfa.image.translate_xy(image,t,0)

#jpeg_quality = 10 #must be between [0,100]
#aug = tf.image.adjust_jpeg_quality(image, jpeg_quality, name=None)



visualize(image,aug)