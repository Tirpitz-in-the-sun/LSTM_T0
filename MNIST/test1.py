# -*- coding: utf-8 -*-

"""
@version: python2.7
@author: Geng Tang
@contact: tanggeng@citics.com
@site: CITICS Beijing
@file: test1.py
@time: 2017/9/18 12:22
Basic Description:

"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':

    # 构建图
    # matrix1 = tf.constant([[3., 3.]])
    # matrix2 = tf.constant([[2.], [2.]])
    # product = tf.matmul(matrix1, matrix2)

    # # 在一个会话中启动图
    # # method 1
    # sess = tf.Session()
    #
    # result = sess.run(product)
    # print result
    #
    # sess.close()
    #
    # # method 2
    # with tf.Session() as sess:
    #     result = sess.run([product])
    #     print result

    # # 指定显卡
    # with tf.Session() as sess:
    #     with tf.device("/gpu:1"):
    #         matrix1 = tf.constant([[3., 3.]])
    #         matrix2 = tf.constant([[2.], [2.]])
    #         product = tf.matmul(matrix1, matrix2)
    #         result = sess.run([product])
    #         print result

    # # 创建一个变量, 初始化为标量 0.
    # state = tf.Variable(0, name="counter")
    #
    # one = tf.constant(1)
    # new_value = tf.add(state, one)
    # update = tf.assign(state, new_value)
    #
    # # 启动图后, 变量必须先经过`初始化` (init) op 初始化,  变量都需要经过初始化操作
    # # 首先必须增加一个`初始化` op 到图中.
    # init_op = tf.initialize_all_variables()
    #
    # # 启动图, 运行 op
    # with tf.Session() as sess:
    #     # 运行 'init' op
    #     sess.run(init_op)
    #     # 打印 'state' 的初始值
    #     print sess.run(state)
    #     # 运行 op, 更新 'state', 并打印 'state'
    #     for _ in range(3):
    #         sess.run(update)
    #         print sess.run(state)

    # input1 = tf.constant(3.0)
    # input2 = tf.constant(2.0)
    # input3 = tf.constant(5.0)
    # intermed = tf.add(input2, input3)
    # mul = tf.multiply(input1, intermed)
    #
    # with tf.Session() as sess:
    #     result = sess.run([mul, intermed])
    #     print result
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)