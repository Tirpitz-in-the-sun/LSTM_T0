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
from LSTM.min_bar_forecasting.data_process.lstm_data_process import *

# 定义变量
HIDDEN_LAYER_UNIT = 20      # 隐藏层的大小
FEATURE_SIZE = 9            # 特征的大小
CLASS_SIZE = 3             # 输出的size
LEARNING_RATE = 0.0006      # 学习率
TIME_STEP = 20               # 时间步长
BATCH_SISE = 100             # 单个batch的规模
EPOCH = 20000               # 训练次数


# ——————————————————定义神经网络变量——————————————————
def lstm(x_input, reuse=None):
    real_batch_size = tf.shape(x_input)[0]
    time_step = tf.shape(x_input)[1]

    # 输入层、输出层权重、偏置
    if reuse==None:
        with tf.variable_scope("myrnn", reuse=None) as scope:
            weights = {
                'in': tf.Variable(tf.random_normal([FEATURE_SIZE, HIDDEN_LAYER_UNIT]), name='weights_in'),
                'out': tf.Variable(tf.random_normal([HIDDEN_LAYER_UNIT, CLASS_SIZE]), name='weights_out')
            }
            biases = {
                'in': tf.Variable(tf.constant(0.1, shape=[HIDDEN_LAYER_UNIT, ]), name='biases_in'),
                'out': tf.Variable(tf.constant(0.1, shape=[CLASS_SIZE, ]), name='biases_out')
            }
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
    input_rnn = tf.matmul(input_of_hidden, w_in)+b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, HIDDEN_LAYER_UNIT])  # 将tensor转成3维，作为lstm cell的输入
    cell = rnn.BasicLSTMCell(HIDDEN_LAYER_UNIT, forget_bias=1.0, state_is_tuple=True)
    init_state = cell.zero_state(real_batch_size, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32,
                                                 time_major=False)
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(final_states[1], w_out)+b_out
    # print pred, final_states
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm():
    x_input = tf.placeholder(tf.float32, shape=[None, TIME_STEP, FEATURE_SIZE])
    y_output = tf.placeholder(tf.float32, shape=[None, CLASS_SIZE])

    pred, _ = lstm(x_input)
    # 损失函数  均方误差
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_output))

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_output, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        index = 0

        # sample_constructor = SampleConstructor(4, 20161231)
        sample_constructor = MultiCodeSampleConsructor([0,1,2,3,4], 20161231)

        # 重复训练EPOCH次
        while index < EPOCH:
            # 随机采样数据样本
            train_data = sample_constructor.construct_batch_data(TIME_STEP, BATCH_SISE, 0.6)

            train_x = train_data[0]
            train_y = train_data[1]
            if np.sum(train_x != train_x) or np.sum(train_y != train_y):
                continue
            _, loss_, accuracy_ = sess.run([train_op, loss, accuracy], feed_dict={x_input: train_x, y_output: train_y})
            index += 1
            if index % 200 == 0:
                print(index, loss_, accuracy_)
                print(u"保存模型：", saver.save(sess, 'stock_min.model', global_step=index))


if __name__ == '__main__':

    train_lstm()