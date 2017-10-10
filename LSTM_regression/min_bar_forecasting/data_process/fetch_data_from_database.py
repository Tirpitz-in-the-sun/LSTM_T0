# -*- coding: utf-8 -*-

"""
@version: python2.7
@author: Geng Tang
@contact: tanggeng@citics.com
@site: CITICS Beijing
@file: fetch_data_from_database.py
@time: 2017/9/27 17:32
Basic Description:

"""
import numpy as np
import os, sys
this_file = os.path.split(os.path.realpath(__file__))[0]
room_root = os.path.realpath(this_file + "/../../../../../../")
print room_root
sys.path.append(room_root)
from autobase import dl
dl.init('/data/cache/stock')

# ——————————————————导入数据——————————————————————
# 分钟数据
intra_close = dl.get_data("intra_close")
intra_open = dl.get_data("intra_open")
intra_volume = dl.get_data("intra_volume")  # 1min成交量
intra_high = dl.get_data("intra_high")
intra_low = dl.get_data("intra_low")
intra_value = dl.get_data("intra_value")
intra_vwap = dl.get_data("intra_vwap")

# 日级别数据
close = dl.get_data('close')
(ma5, ma20, ma60) = dl.eval_expr(['ma(close,5)', 'ma(close,20)', 'ma(close, 60)'], delay=1, universe="ALL")


# 截取特定股票的数据
def data_cut(code_list):

    ii = [dl.get_ii(code) for code in code_list]
    dates = dl.dates
    save_path = this_file + '/intra/'

    # 保存分钟线数据
    np.save(save_path + "intra_close.npy", intra_close[:, :, ii])
    np.save(save_path + "intra_open.npy", intra_open[:, :, ii])
    np.save(save_path + "intra_volume.npy", intra_volume[:, :, ii])
    np.save(save_path + "intra_high.npy", intra_high[:, :, ii])
    np.save(save_path + "intra_low.npy", intra_low[:, :, ii])
    np.save(save_path + "intra_value.npy", intra_value[:, :, ii])
    np.save(save_path + "intra_vwap.npy", intra_vwap[:, :, ii])

    # 保存日线数据
    save_path = 'day/'
    np.save(save_path + 'dates.npy', dates)
    np.save(save_path + 'close.npy', close[:, ii])
    np.save(save_path + 'ma5.npy', ma5[:, ii])
    np.save(save_path + 'ma20.npy', ma20[:, ii])
    np.save(save_path + 'ma60.npy', ma60[:, ii])


if __name__ == '__main__':
    code_list = [u'600519.SH', u'600660.SH', u'000002.SZ', u'000651.SZ', u'000839.SZ', u'600089.SH',
                 u'600050.SH', u'600031.SH', u'601933.SH', u'600795.SH', u'600135.SH', u'000089.SZ']
    data_cut(code_list)