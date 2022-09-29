#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:42:19 2018

@author: reza
"""

#logistic regression
import tensorflow as tf
import numpy as np

w=tf.Variable(tf.random_normal([2,1]))
b=tf.Variable(0.)
#1/1+e^-(teta^T*x)
def comput(X):
    return tf.add(tf.matmul(X,w),b)#/x*teta+b
def inference(X):
    return tf.sigmoid(comput(X))
def loss(X,Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=comput(X),labels=Y))
def inputs():
    email=[[1,1],[1,2],[2,3],[3,2],[1,4],[1,5],[2,6]]
    spam=[[0],[0],[1],[0],[0],[1],[1]]
    return tf.to_float(email),tf.to_float(spam)
def train(Total_loss):
    learning_rate=0.001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(Total_loss)
def evaluation(sess,X,Y):
    p=sess.run(inference([[1.,1.]]))
    if p.round()==1:
        print("pos")
    else:
        print("neg")
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X,Y=inputs()
    Total=loss(X,Y)
    train_op=train(Total)
    coord=tf.train.Coordinator()
    threeds=tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(2000):
        sess.run([train_op])
        if i%10==0:
            print("loss:",sess.run([Total]))
    evaluation(sess,X,Y)
    coord.request_stop()
    coord.join(threeds)
    
    

