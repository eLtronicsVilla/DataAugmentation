# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 06:41:21 2020

@author: brije
"""


import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

'''#tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


# Place tensors on the GPU
#with tf.device("/GPU:0"):
#  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)
print(c)


#tf.debugging.set_log_device_placement(True)'''

try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)
except RuntimeError as e:
  print(e)



