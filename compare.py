#!/usr/bin/python3

import sys

import random
import math 
import numpy as np
import tensorflow as tf
from time import time


def python_(x1, x2):
    m = len(x1)
    total = 0
    for i in range(m):
        total += math.sqrt(x1[i] ** 2 + x2[i] ** 2)
    return total/m
    
    
def numpy_(x1, x2):
    avg = np.mean(np.sqrt(x1 ** 2 + x2 ** 2))
    return avg

def tensorflow_(x1, x2):
  m = len(x1)
  x1_ = tf.placeholder("float64", [m])
  x2_ = tf.placeholder("float64", [m])
  distances = tf.sqrt(tf.square(x1_) + tf.square(x2_))
  avg = tf.reduce_mean(distances)
  
  with tf.Session() as sess:
    avg_ = sess.run(avg, feed_dict={x1_: x1, x2_: x2})
    return avg_
    
    
if __name__ == "__main__":
    N = 10 ** 6
    if len(sys.argv) > 1:
      N = int(sys.argv[1])
    print("Total nunber of records: ", N)
    
    random.seed(1)
    x1 = [random.random() for i in range(N)]
    x2 = [random.random() for i in range(N)]
    #print("x1", x1)
    #print("x2", x2)

    start = time()
    result = python_(x1, x2)
    print("Python result", result, "time: ", time() - start, " ms")


    x1_np = np.array(x1, dtype = np.float64)
    x2_np = np.array(x2)
    
    start = time()
    result = numpy_(x1_np, x2_np)
    print("Numpy result", result, "time: ", time() - start, " ms")
    
    
    start = time()
    result = tensorflow_(x1, x2)
    print("Tensorflow result", result, "time: ", time() - start, " ms")
    