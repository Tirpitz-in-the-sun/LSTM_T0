#coding=utf-8
"""
@version: python2.7
@author: Geng Tang
@contact: tanggeng@citics.com
@site: CITICS Beijing
@file: test1.py
@time: 2017/9/18 12:22
Basic Description:

"""
import tensorflow as tf
from tensorflow.contrib import rnn
from LSTM.min_bar_forecasting.data_process.lstm_data_process_single_stock import *

# 定义变量
HIDDEN_LAYER_UNIT = 120      # 隐藏层的大小
FEATURE_SIZE = 9            # 特征的大小
OUTPUT_SIZE = 1             # 输出的size
LEARNING_RATE = 0.0006      # 学习率
TIME_STEP = 20               # 时间步长
BATCH_SISE = 100             # 单个batch的规模
EPOCH = 100000               # 训练次数


# ——————————————————定义神经网络变量——————————————————
def lstm(x_input, reuse=None):
    real_batch_size = tf.shape(x_input)[0]
    time_step = tf.shape(x_input)[1]

    # 输入层、输出层权重、偏置
    if reuse==None:
        with tf.variable_scope("myrnn", reuse=None) as scope:
            weights = {
                'in': tf.Variable(tf.random_normal([FEATURE_SIZE, HIDDEN_LAYER_UNIT]), name='weights_in'),
                'out': tf.Variable(tf.random_normal([HIDDEN_LAYER_UNIT, 1]), name='weights_out')
            }
            biases = {
                'in': tf.Variable(tf.constant(0.1, shape=[HIDDEN_LAYER_UNIT, ]), name='biases_in'),
                'out': tf.Variable(tf.constant(0.1, shape=[1, ]), name='biases_out')
            }

            w_in = weights['in']
            b_in = biases['in']
            input_of_hidden = tf.reshape(x_input, [-1, FEATURE_SIZE])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
            input_rnn = tf.matmul(input_of_hidden, w_in)+b_in
            input_rnn = tf.reshape(input_rnn, [-1, time_step, HIDDEN_LAYER_UNIT])  # 将tensor转成3维，作为lstm cell的输入
            cell = rnn.BasicLSTMCell(HIDDEN_LAYER_UNIT)
            init_state = cell.zero_state(real_batch_size, dtype=tf.float32)
            # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果

            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    else:
        with tf.variable_scope("myrnn", reuse=True) as scope:
            weights = {
                'in': tf.global_variables()[0],
                'out': tf.global_variables()[1]
            }
            biases = {
                'in': tf.global_variables()[2],
                'out': tf.global_variables()[3]
            }

            w_in = weights['in']
            b_in = biases['in']
            input_of_hidden = tf.reshape(x_input, [-1, FEATURE_SIZE])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
            input_rnn = tf.matmul(input_of_hidden, w_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, time_step, HIDDEN_LAYER_UNIT])  # 将tensor转成3维，作为lstm cell的输入
            cell = rnn.BasicLSTMCell(HIDDEN_LAYER_UNIT)
            init_state = cell.zero_state(real_batch_size, dtype=tf.float32)
            # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
            # output_rnn: (batch_size, time_step, hidden_size)
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)

    output = tf.reshape(final_states[1], [-1, HIDDEN_LAYER_UNIT])   # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out)+b_out
    # print pred, final_states
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm():
    x_input = tf.placeholder(tf.float32, shape=[None, TIME_STEP, FEATURE_SIZE])
    y_output = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

    pred, _ = lstm(x_input)
    # 损失函数  均方误差
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1])-tf.reshape(y_output, [-1])))

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        index = 0

        sample_constructor = SampleConstructor(4, 20161231)

        # 重复训练EPOCH次
        while index < EPOCH:
            # 随机采样数据样本
            train_data = sample_constructor.construct_batch_data(TIME_STEP, BATCH_SISE, 0.5)
            train_x = train_data[0]
            train_y = train_data[1]
            if np.sum(train_x != train_x) or np.sum(train_y != train_y):
                continue
            _, loss_ = sess.run([train_op, loss], feed_dict={x_input: train_x, y_output: train_y})
            index += 1
            if index % 200 == 0:
                print(index, loss_)
                print(u"保存模型：", saver.save(sess, 'stock_min.model', global_step=index))


if __name__ == '__main__':

    train_lstm()