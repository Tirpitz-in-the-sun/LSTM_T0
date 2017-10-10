# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:23:31 2017

@author: Wuge
"""
import random
import os
import sys
import pandas as pd
import numpy as np
from collections import OrderedDict
# ——————————————————导入数据——————————————————————
this_file = os.path.split(os.path.realpath(__file__))[0]


intra_fold = this_file + '/intra/'
day_fold = this_file + '/day/'

intra_close = np.load(intra_fold + "intra_close.npy")
intra_open = np.load(intra_fold + "intra_open.npy")
intra_volume = np.load(intra_fold + "intra_volume.npy")
# intra_value = np.load(intra_fold + "intra_value.npy")
intra_high = np.load(intra_fold + "intra_high.npy")
intra_low = np.load(intra_fold + "intra_low.npy")
intra_vwap = np.load(intra_fold + "intra_vwap.npy")


intra_price_keys = ['intra_close', 'intra_high', 'intra_low', 'intra_open', 'intra_vwap']
intra_non_price_keys = ['intra_value']

intra_close[intra_close == 0] = float('nan')
intra_low[intra_low == 0] = float('nan')
intra_high[intra_high == 0] = float('nan')
intra_open[intra_open == 0] = float('nan')
intra_vwap[intra_vwap == 0] = float('nan')


dates = np.load(day_fold + "dates.npy")
pre_close = np.load(day_fold + "pre_close.npy")
close = np.load(day_fold + "close.npy")
ma5 = np.load(day_fold + "ma5.npy")
ma20 = np.load(day_fold + "ma20.npy")
ma60 = np.load(day_fold + "ma60.npy")

daily_price_keys = ['ma5', 'ma20', 'ma60']


class SampleConstructor(object):

    def __init__(self, code, train_end_date):
        # 定义变量
        self.code = code
        if self.code < 0:
            raise Exception(u"%s is not a stock code" % code)
        self.train_end_date_index = np.argmin(np.abs(dates-train_end_date))
        self.init_data = OrderedDict()

        # 考虑到深市，最后三个数据不要，集合竞价
        self.init_data['intra_close'] = intra_close[:, 0:237, self.code]
        self.init_data['intra_low'] = intra_low[:, 0:237, self.code]
        self.init_data['intra_high'] = intra_high[:, 0:237, self.code]
        self.init_data['intra_open'] = intra_open[:, 0:237, self.code]
        self.init_data['intra_volume'] = intra_volume[:, 0:237, self.code]
        self.init_data['intra_vwap'] = intra_vwap[:, 0:237, self.code]
        self.init_data['intra_value'] = self.init_data['intra_vwap'] * self.init_data['intra_volume']

        # 日间原始数据
        self.init_data['close'] = close[:, self.code]
        self.init_data['pre_close'] = np.array([0] + list(close[0:len(self.init_data['close'])-1, self.code]))
        self.init_data['ma5'] = ma5[:, self.code]
        self.init_data['ma20'] = ma20[:, self.code]
        self.init_data['ma60'] = ma60[:, self.code]

        self.train_init_data = OrderedDict()
        self.test_init_data = OrderedDict()

        self.price_mean = 0
        self.price_std = 0
        self.value_mean = 0
        self.value_std = 0

        self.available_dates_array = []
        self.not_available_dates_array = []
        self.available_train_dates_array = []
        self.available_test_dates_array = []

        self.standard_data_input = np.array([])
        self.standard_data_output = np.array([])
        self.volatility_output = []
        self.standard_data_output_dict = OrderedDict()
        self.pos_large_vol_index = []
        self.small_vol_index = []
        self.neg_large_vol_index = []

        self.pre_close_test_sample = []
        self.__data_clean()

    def __data_clean(self):

        # 选择能够用于训练的minbar
        available_dates = []
        not_available_dates = []

        for i in range(self.init_data['intra_close'].shape[0]):
            # 去除无前收盘价的日期
            if self.init_data['pre_close'][i] == 0:
                not_available_dates.append(i)
                continue
            # 去除异常数据
            if np.min(self.init_data["intra_vwap"][i]) < 0.1:
                not_available_dates.append(i)
                continue
            # 整理intra_vwap中的nan，用1/2（high+low）
            intra_vwap_nan_where = np.where(self.init_data["intra_vwap"][i] != self.init_data["intra_vwap"][i])
            self.init_data["intra_vwap"][i][intra_vwap_nan_where] = 0.5 * (self.init_data["intra_high"][i] +
                                                                           self.init_data["intra_low"][i])[intra_vwap_nan_where]
            # 去除除权日
            daily_gain = self.init_data['close'][i]/self.init_data['pre_close'][i]
            if daily_gain > 1.11 or daily_gain < 0.89:
                not_available_dates.append(i)
                continue
            # 去除数据中含有错误的日期
            na_num = 0
            for key in intra_price_keys + daily_price_keys:
                na_num += (int(np.sum(self.init_data[key][i] != self.init_data[key][i]))
                           + int(np.sum(self.init_data[key][i] == 0 )))
            if na_num != 0:
                not_available_dates.append(i)
                continue

            if np.count_nonzero(self.init_data['intra_volume'][i] == 0) <= 150:
                available_dates.append(i)
            else:
                not_available_dates.append(i)
        self.available_dates_array = np.array(available_dates)
        self.not_available_dates_array = np.sort(not_available_dates)
        self.available_train_dates_array = self.available_dates_array[self.available_dates_array
                                                                      <= self.train_end_date_index]
        self.available_test_dates_array = self.available_dates_array[self.available_dates_array
                                                                     > self.train_end_date_index]
        # 把intra_value中的nan重置
        self.init_data['intra_value'] = self.init_data['intra_vwap'] * self.init_data['intra_volume']
        pre_close_base = np.transpose(self.init_data['pre_close'][np.newaxis])

        # 日内价格转除去股价因素
        for key in intra_price_keys:
            self.init_data[key] /= pre_close_base

        # daily价格数据除去股价因素
        for key in daily_price_keys:
            self.init_data[key] /= self.init_data['pre_close']

        # 可用数据的映射
        for key in self.init_data.keys():
            self.train_init_data[key] = self.init_data[key][self.available_train_dates_array]
            self.test_init_data[key] = self.init_data[key][self.available_test_dates_array]

        print u'训练数据的价格已去除股价因素'

        # Zscore标准化操作, 标准化的基准从训练数据中获取
        self.price_mean = np.mean(self.train_init_data['intra_close'])
        self.price_std = np.sqrt(np.var(self.train_init_data['intra_close']))

        for key in intra_price_keys + daily_price_keys:
            self.train_init_data[key] = (self.train_init_data[key] - self.price_mean)/self.price_std
            self.test_init_data[key] = (self.test_init_data[key] - self.price_mean) / self.price_std

            # 对price数据进行双边的winsorize
            [left_value, right_value] = np.percentile(self.train_init_data[key], [1, 99])
            self.train_init_data[key][self.train_init_data[key] > right_value] = right_value
            self.train_init_data[key][self.train_init_data[key] < left_value] = left_value
            self.test_init_data[key][self.test_init_data[key] > right_value] = right_value
            self.test_init_data[key][self.test_init_data[key] < left_value] = left_value
        # 对value取对数
        self.train_init_data["intra_value"] = np.log(self.train_init_data["intra_value"] + 1)
        self.test_init_data["intra_value"] = np.log(self.test_init_data["intra_value"] + 1)
        # 对value数据进行单边的winsorize
        right_value = np.percentile(self.train_init_data["intra_value"], 99)
        self.train_init_data["intra_value"][self.train_init_data["intra_value"] > right_value] = right_value
        self.test_init_data["intra_value"][self.test_init_data["intra_value"] > right_value] = right_value


        self.value_mean = np.mean(self.train_init_data["intra_value"])
        self.value_std = np.sqrt(np.var(self.train_init_data["intra_value"]))

        self.train_init_data['intra_value'] = (self.train_init_data['intra_value'] - self.value_mean)/self.value_std
        self.test_init_data['intra_value'] = (self.test_init_data['intra_value'] - self.value_mean)/self.value_std
        print np.max(self.train_init_data['intra_value']), np.min(self.train_init_data['intra_value'])

    def __construct_standard_data(self, time_step, usage='train', sday=0):

        if usage == 'train':
            data_set = self.train_init_data
            available_index = range(data_set['intra_close'].shape[0])
        else:
            if sday == 0:
                data_set = self.test_init_data
                available_index = range(data_set['intra_close'].shape[0])
            else:
                date_index = np.where(dates == sday)[0][0]
                if date_index in self.available_test_dates_array:
                    final_index = np.where(self.available_test_dates_array == date_index)[0][0]
                    data_set = self.test_init_data
                    available_index = [final_index]
                else:
                    raise Exception(u"%d is not a trade date" % sday)

        standard_data_input = []
        standard_data_output = []
        volatility_output = []
        vola_standard = 0.008  # 打败交易费
        for t in range(data_set['intra_close'].shape[0]):
            if t not in available_index:
                continue
            if usage == 'test':
                trading_date = dates[self.available_test_dates_array[t]]
                pre_close_date = self.init_data["pre_close"][self.available_test_dates_array[t]]
            else:
                trading_date = dates[self.available_train_dates_array[t]]
                pre_close_date = self.init_data["pre_close"][self.available_train_dates_array[t]]

            self.pre_close_test_sample.append(pre_close_date)
            self.standard_data_output_dict[trading_date] = OrderedDict()
            for z in range(237-time_step):
                train_temp = []
                for key in daily_price_keys:
                    train_temp.append(np.ones(shape=time_step)*data_set[key][t])

                for key in intra_price_keys + intra_non_price_keys:
                    train_temp.append(data_set[key][t,  z:(z+time_step)])

                input_data = np.array(train_temp).T
                standard_data_input.append(input_data)
                output_data = data_set['intra_close'][t, z + time_step]

                pre_data = data_set['intra_close'][t, z + time_step-1]
                if z == 0:
                    self.standard_data_output_dict[trading_date][z + time_step - 1] = (pre_data*self.price_std +
                                                                                       self.price_mean)*pre_close_date
                self.standard_data_output_dict[trading_date][z + time_step] = (output_data*self.price_std +
                                                                               self.price_mean)*pre_close_date
                volatility = (output_data-pre_data)/(pre_data+self.price_mean/self.price_std)
                volatility_output.append(volatility)
                if volatility >= vola_standard:
                    standard_data_output.append([1, 0, 0])
                elif volatility <= -vola_standard:
                    standard_data_output.append([0, 0, 1])
                else:
                    standard_data_output.append([0, 1, 0])

        self.standard_data_input = np.array(standard_data_input)
        # self.standard_data_output = np.array(standard_data_output)
        self.volatility_output = np.array(volatility_output)

        self.pos_large_vol_index = np.where(self.volatility_output >= vola_standard)[0]
        self.small_vol_index = np.where((self.volatility_output < vola_standard)*(self.volatility_output > - vola_standard))[0]
        self.neg_large_vol_index = np.where(self.volatility_output <= -vola_standard)[0]

        self.standard_data_output = np.array(standard_data_output)

    def construct_batch_data(self, time_step, batch_size, large_volatility_prop=0.5):

        if self.standard_data_output.shape[0] == 0:
            self.__construct_standard_data(time_step)
        else:
            pass

        train_x = self.standard_data_input
        train_y = self.standard_data_output
        large_batch_size = int(batch_size * large_volatility_prop)
        pos_large_batch_size = large_batch_size/2
        neg_large_batch_size = large_batch_size-pos_large_batch_size
        small_batch_size = batch_size - large_batch_size
        index_pos_large = random.sample(self.pos_large_vol_index, pos_large_batch_size)
        index_neg_large = random.sample(self.neg_large_vol_index, neg_large_batch_size)
        index_small = random.sample(self.small_vol_index, small_batch_size)
        index = index_pos_large + index_neg_large + index_small
        x_result = train_x[index]
        y_result = train_y[index]
        return [x_result, y_result]

    def construct_test_sday(self, time_step, sday):

        self.__construct_standard_data(time_step, 'test', sday)
        x_result = self.standard_data_input
        y_result = self.standard_data_output

        # Zscore逆标准化
        y_result = y_result*self.price_std + self.price_mean

        y_result = y_result.reshape(-1)
        return [x_result, y_result]


if __name__ == '__main__':

    symbol = 10
    sample_constructor = SampleConstructor(symbol, 20161231)
    # all_value = sample_constructor.train_init_data['intra_value'].reshape(-1)
    # all_volume = sample_constructor.train_init_data['intra_volume'].reshape(-1)
    # pd.DataFrame(all_value).to_csv("intra_value_%d.csv" % symbol)
    # pd.DataFrame(all_volume).to_csv("intra_volume_%d.csv" % symbol)
    sample_constructor.construct_batch_data(15, 100, 0.6)
    # sample_constructor.construct_test_sday(15, 0)
