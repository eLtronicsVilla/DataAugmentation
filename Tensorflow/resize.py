# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:08:01 2020

@author: eLtronicavilla
"""

##############################################################################
# Resizing :- 
# Input - Image as tensor of several types.
# Output - always resized image as float32
# It support both 3D and 4D image as batches of input and output.
# 4D for batches of image and 3D tensor for indivisual image.
# Resized image will be distarted if original aspect ratio is not same in size.

# tf.image.resize() 

##############################################################################

#import the library
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np


# create dataset

# Note : tf.keras.utils.get_files -  api get_files download a file from the url if not in the cache using utility package of keras library 
#image_path = tf.keras.utils.get_file("cat.jpg","https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")
# python image library to open the image 
image = PIL.Image.open("E:\\workspace\\workspace-brijesh\\daily_dose\\TF-dose\\tf_image\\cat.png")


# io package is used to decode the image
# read and output the contain of input file 
#image_string = tf.io.read_file(image_path)
# decode the image in jpg
#image = tf.image.decode_png(image_string)


def visualize(original,augmented):
    plt.subplot(1,2,1)
    plt.title('input')
    plt.imshow(original)
    plt.subplot(1,2,2)
    plt.title('out')
    # converting image datatype to float in range [0,255]
    plt.imshow(augmented/255.0)
    
    
    
# resize image to size in specifying method.
# resized method is bilinear with preserving aspect ratio.
image = tf.convert_to_tensor(np.array(image))
exp_img = tf.expand_dims(image, axis=0)
resized_img = tf.image.resize(exp_img,[100,100],method='bilinear',preserve_aspect_ratio=False,antialias=False,name='resize')
resized_img = tf.squeeze(resized_img, axis=0)

visualize(image,resized_img)
