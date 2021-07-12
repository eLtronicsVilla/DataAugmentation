# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import the library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#tf.enable_eager_execution()

# prepare the dataset
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
dataset = tf.data.Dataset.from_tensor_slices((x_train[0:8] / 255).astype(np.float32))

# plot the image

def plot_images(dataset,n_images,sample_per_images):
    output = np.zeros((32*n_images,32*sample_per_images,3))
    
    row = 0
    
    for images in dataset.repeat(sample_per_images).batch(n_images):
        output[:,row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1
        
    plt.figure()
    plt.imshow(output)
    plt.show()
    
# implementing Augmentation

def augment(x: tf.Tensor) -> tf.Tensor:
    return x

def rotate(x:tf.Tensor) -> tf.Tensor:
    # rotate 0,90,180,270 degree
    return tf.image.rot90(x, tf.random.uniform(shape=[],minval=0,maxval=4,dtype=tf.int32))

def flip(x:tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x

def color (x:tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x,0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x,0.05)
    x = tf.image.random_contrast(x,0.7,1.3)
    return x


def zoom(x: tf.Tensor) -> tf.Tensor:
    # generate 20 crop settings: ranging 1% to 20% crop.
    scales = list(np.arange(0.8,1,0.01))
    boxes = np.zeros((len(scales),4))
    
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 *scale)
        x2 = y2 = 0.5 + (0.5 *scale)
        boxes[i] = [x1,y1,x2,y2]
        
    def random_crop(img):
        # different crop from image
        crops = tf.image.crop_and_resize([img], boxes= boxes, box_indices = np.zeros(len(scales)), crop_size=(32,32))
        # return a random crop
        return crops[tf.random.uniform(shape = [],minval=0,maxval=len(scales),dtype=tf.int32)]
    
    choice = tf.random.uniform(shape=[],minval = 0.0,maxval = 1.0,dtype=tf.float32)
    
    # apply cropping 50% of the time
    return tf.cond(choice < 0.5,lambda: x, lambda:random_crop(x))


# Augmenting with dataset
# all defined function we can combine them into a single pipeline.
# applying this function to a tensorflow dataset is very easy using map() function.
# map function takes a function and return augmented dataset
# when the new dataset is evaluate this , function defined will be apply for all element in the set.

# Note : to drastically increase the speed of these operation -  execute them in parallel, TF operation support this.
# tf.Data - api can be used wth num_parallel_calls parameters of map()  function.
# it is advisable to set these parameter =  number of CPU available.

# Note : some of these operation can result in image that have value outside the normal range of [0,1]
# Make sure that these range are not exceeded a cliping function such as tf.clip_by_value is recommended.

# Add Augmentation

aug =  [flip,color,zoom ,rotate]

# Add the augmentation to the dataset
for f in aug:
    # apply the aug run job in parallel
    dataset = dataset.map(f,num_parallel_calls=4)

# make sure that value are still in [0,1]
dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=4)

plot_images(dataset, n_images=10, sample_per_images=15)    