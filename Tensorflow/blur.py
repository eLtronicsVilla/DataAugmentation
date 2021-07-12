# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:48:13 2020

@author: eLtronicsvilla
"""
 
#import tensorflow.compat.v1 as tf 

#from tensorflow.compat.v1 import enable_eager_execution
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import PIL

import matplotlib.pyplot as plt



def gaussian_blur(image, kernel_size, sigma, padding='SAME'): 
   """Blurs the given image with separable convolution. 
  
  
   Args: 
     image: Tensor of shape [height, width, channels] and dtype float to blur. 
     kernel_size: Integer Tensor for the size of the blur kernel. This is should 
       be an odd number. If it is an even number, the actual kernel size will be 
       size + 1. 
     sigma: Sigma value for gaussian operator. 
     padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'. 
  
   Returns: 
     A Tensor representing the blurred image. 
   """ 
   #radius =  tf.to_int32(kernel_size / 2)
   #radius = tf.convert_to_tensor(kernel_size / 2,dtype=tf.float32)
   radius =  tf.dtypes.cast(kernel_size / 2,dtype=tf.float32)
   kernel_size = radius * 2 + 1 
   #x = tf.to_float(tf.range(-radius, radius + 1))
   x = tf.dtypes.cast(tf.range(-radius, radius + 1),tf.float32)
   #x = tf.convert_to_tensor(tf.range(-radius, radius + 1),dtype=tf.float32)
   blur_filter = tf.exp( -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.dtypes.cast(sigma,tf.float32), 2.0))) 
   #blur_filter = tf.exp( -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.convert_to_tensor(sigma,tf.float32), 2.0)))
   blur_filter /= tf.reduce_sum(blur_filter) 
   # One vertical and one horizontal filter. 
   blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1]) 
   blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1]) 
   num_channels = tf.shape(image)[-1] 
   blur_h = tf.tile(blur_h, [1, 1, num_channels, 1]) 
   blur_v = tf.tile(blur_v, [1, 1, num_channels, 1]) 
   
   expand_batch_dim = image.shape[2] == 3 
   if expand_batch_dim: 
     # Tensorflow requires batched input to convolutions, which we can fake with 
     # an extra dimension. 
     image = tf.expand_dims(image, axis=0) 
   blurred = tf.nn.depthwise_conv2d( image, blur_h, strides=[1, 1, 1, 1], padding=padding) 
   blurred = tf.nn.depthwise_conv2d( blurred, blur_v, strides=[1, 1, 1, 1], padding=padding) 
   if expand_batch_dim: 
     blurred = tf.squeeze(blurred, axis=0)
     #print(type(blurred))
   return blurred 

def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  #plt.title('Original image')
  #plt.imshow(original)
  plt.imshow(original)
  plt.subplot(1,2,2)
  plt.title('Augmented image')
  #aug = np.expand_dims(np.asarray(augmented).astype(np.float32), axis =0 )
  #plt.imshow(tf.dtypes.cast(augmented,tf.uint8))
  #aug = tf.cast(augmented, tf.uint8)
  #plt.imshow(np.array(augmented))
  #plt.imshow(augmented)
  plt.show()


if __name__ == '__main__':
    
# =============================================================================
#     filenames = ['input.jpg']
#     cur_dir = os.getcwd()
#     filename_queue = tf.train.string_input_producer(filenames)
# 
#     reader = tf.WholeFileReader()
#     key, value = reader.read(filename_queue)
# 
#     images = tf.image.decode_jpeg(value, channels=3)
# =============================================================================
    image_path = tf.keras.utils.get_file("cat.jpg", "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")
    #image = PIL.Image.open(image_path)
    #im = PIL.Image.open("input.jpg")
    #im = tf.keras.preprocessing.image.load_img( "./input.jpg", grayscale=False, color_mode='rgb')
    #image = tf.convert_to_tensor(im, dtype=tf.float32)
    #image  = tf.keras.preprocessing.image.img_to_array(image )
    #image_path = tf.keras.utils.
    image_string=tf.io.read_file(image_path)
    image=tf.image.decode_jpeg(image_string,channels=3)
    
    #image = tf.cast(im, tf.float32)
    #image = tf.convert_to_tensor(image, dtype=tf.float32)

    
    

    sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
    gb = gaussian_blur( image, kernel_size=5, sigma=sigma, padding='SAME') 
    #gb = tfa.image.median_filter2d(image,[3,3])
    #img = Image.fromarray(gb, "RGB")
    #img = tf.make_ndarray(gb)
    #proto_t = tf.make_tensor_proto(gb)
    #gb = tf.convert_to_tensor(gb, dtype=tf.float32)
    #gb = np.array(gb)
    #arrr = np.squeeze(arr)
    #gb = tf.keras.preprocessing.image.array_to_img(gb)
    #gb = PIL.Image.fromarray()
    
    #img = tf.make_ndarray(gb)
   
    print(type(gb))
    print(tf.shape(gb))
    gb = tf.cast(gb, tf.int64)
    print(type(image))
    #print(type(img))
    visualize(image, gb)
    







