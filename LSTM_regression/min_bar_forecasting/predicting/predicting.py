# -*- coding: utf-8 -*-

"""
@version: python2.7
@author: Geng Tang
@contact: tanggeng@citics.com
@site: CITICS Beijing
@file: Predicting.py
@time: 2017/9/21 11:28
Basic Description:

"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LSTM.min_bar_forecasting.data_process.lstm_data_process_single_stock import *
from LSTM.min_bar_forecasting.training.training import lstm
from LSTM.min_bar_forecasting.training.training import tf

this_file = os.path.split(os.path.realpath(__file__))[0]
room_root = os.path.realpath(this_file + "/../")
TIME_STEP = 10              # 时间步长
FEATURE_SIZE = 9            # 特征的大小


# ————————————————预测模型————————————————————
def prediction(symbol=0, date=20170828, reuse=None, module_file_='6000'):

    sample_constructor = SampleConstructor(symbol, 20161231)

    test_x, test_y = sample_constructor.construct_test_sday(TIME_STEP, date)

    x_input = tf.placeholder(tf.float32, shape=[None, TIME_STEP, FEATURE_SIZE], name='x_input')
    pred, _ = lstm(x_input, reuse)
    saver = tf.train.Saver(tf.global_variables())
    mean = sample_constructor.price_mean
    std = sample_constructor.price_std

    with tf.Session() as sess:
        # 参数恢复
        module_file = room_root + '/training/stock_min.model-' + module_file_
        try:
            saver.restore(sess, module_file)
        except:
            return
        prob_output = sess.run(pred, feed_dict={x_input: test_x})
        test_predict = prob_output.reshape((-1))
        test_predict = np.array(test_predict)*std+mean

    return test_predict, test_y, sample_constructor


# test_y 基于test_y本身
def back_test(symbol, test_predict_all, test_y_all, sample_constructor):

    pre_close_test_sample = sample_constructor.pre_close_test_sample
    test_y_all2 = sample_constructor.standard_data_output_dict
    trading_fee = 0.001
    date_len = len(pre_close_test_sample)
    min_bar_len = len(test_predict_all)/date_len
    date_index = 0
    balance_sheet = [0]

    test_y_index = 0
    for pre_close_date in pre_close_test_sample:

        trigger = 0.002*pre_close_date
        trading_datem, test_y2 = test_y_all2.items()[test_y_index]
        test_y_index += 1
        test_y2 = np.array([test_y2[key] for key in test_y2][1:])

        test_y = test_y_all[date_index * min_bar_len:(date_index + 1) * min_bar_len] * pre_close_date

        if np.average(abs(test_y2 - test_y)) > 0.005:
            print symbol, trading_datem
            raise Exception("wrong")

        if symbol == 10 and trading_datem == 20170405:
            print symbol
        test_predict = test_predict_all[date_index * min_bar_len:(date_index + 1) * min_bar_len] * pre_close_date
        date_index += 1
        test_predict_long_sign = test_predict[1:]*(1-trading_fee) - test_y[:len(test_y)-1]*(1+trading_fee)
        test_predict_long_sign[test_predict_long_sign < trigger] = 0
        test_predict_long_sign = np.sign(test_predict_long_sign)
        test_predict_long_sign[test_y[:len(test_y) - 1] >= 1.095 * pre_close_date] = 0
        test_predict_long_sign[test_y[:len(test_y) - 1] <= 0.905 * pre_close_date] = 0

        test_predict_short_sign = -test_predict[1:]*(1+trading_fee) + test_y[:len(test_y)-1]*(1-trading_fee)
        test_predict_short_sign[test_predict_short_sign < trigger] = 0
        test_predict_short_sign = -np.sign(test_predict_short_sign)
        test_predict_short_sign[test_y[:len(test_y) - 1] >= 1.095 * pre_close_date] = 0
        test_predict_short_sign[test_y[:len(test_y) - 1] <= 0.905*pre_close_date] = 0

        test_predict_pos = test_predict_long_sign+test_predict_short_sign

        long_where = np.where(test_predict_long_sign == 1)
        long_profit = (test_y[1:][long_where]*(1-trading_fee) - test_y[:len(test_y)-1][long_where]*(1+trading_fee))\
                      / (test_y[:len(test_y)-1][long_where]*(1+trading_fee))

        short_where = np.where(test_predict_short_sign == -1)
        short_profit = (-test_y[1:][short_where]*(1+trading_fee) + test_y[:len(test_y)-1][short_where]*(1-trading_fee))\
                      / (test_y[1:][short_where]*(1+trading_fee))

        total_profit = float(np.sum(long_profit)) + float(np.sum(short_profit))

        balance_sheet.append(total_profit)

        if total_profit < -0.02:
            fig = plt.figure()
            plt.plot(list(range(len(test_y))), test_y, color='b')
            plt.plot(list(range(len(test_predict))), test_predict, color='r')
            fig.savefig('stock_%d_%d.png' % (symbol, trading_datem))
            print symbol, trading_datem

    balance_sheet = np.cumsum(balance_sheet)
    print balance_sheet
    # 以折线图表示结果
    fig = plt.figure()
    plt.plot(list(range(len(balance_sheet))), balance_sheet, color='b')
    fig.savefig('stock_%d.png' % symbol)


def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".png"):
                os.remove(os.path.join(root, name))


if __name__ == '__main__':

    del_files(this_file)

    index = 0

    for symbol in range(12):
        date = 0
        module_num = '100000'
        if index == 0:
            test_predict, test_y, sample_constructor = \
                prediction(int(symbol), int(date), None, module_num)
            index += 1

        else:
            test_predict, test_y, sample_constructor = \
                prediction(int(symbol), int(date), True, module_num)

        back_test(int(symbol), test_predict, test_y, sample_constructor)


