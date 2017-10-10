# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:24:36 2017

@author: Wuge
"""


'''

此处参数dates 为需要回测的date的array
函数输出my_signals[i]=[date,signals]
date为回测的具体日期   signals是list,结构如下：
signals[i]为该日第i个交易信号
signals[i] = [交易时间，交易价格，flag,position,收益]
交易时间为当日的第i分钟(与test_y对应)

交易价格为交易时点的收盘价，比如min=5结束时，基于对min=6的预测开仓，则交易价格为min=5时的收盘价

flag为交易类型标签
flag=1 买入开仓
flag=2 卖出平仓
flag=3 卖出开仓
flag=4 买入平仓
flag=5 买入平仓，同时反向买入开仓
flag=6 卖出平仓，同时反向卖出开仓

position 该信号发出后持仓状态
0 = 无仓位  1=多 2=空

收益为本笔交易扣除手续费后盈亏，注意，只有开仓的交易存在收益，平仓交易的收益为‘nan’
'''

def learn_signals(symbol=0, dates, reuse=None, module_file_='10000'):

    sample_constructor = SampleConstructor(symbol, TIME_STEP, 0.9)

    x_input = tf.placeholder(tf.float32, shape=[None, TIME_STEP, FEATURE_SIZE], name='x_input')
    pred, _ = lstm(x_input, reuse)
    saver = tf.train.Saver(tf.global_variables())
    mean = reconstruct_parameter[0]
    std = reconstruct_parameter[1]

    with tf.Session() as sess:
        # 参数恢复
        module_file = 'stock_min.model-' + module_file_
  #      module_file = 'C:/Users/Wuge/Desktop/lstm/stock_min.model-' + module_file_
        try:
            saver.restore(sess, module_file)
        except:
            return
        my_signals = []
        for date in dates.tolist():
            test_x, test_y, reconstruct_parameter, _ = sample_constructor.get_day_test(date)
            test_predict = []
            for step in range(len(test_x)):
                prob_output = sess.run(pred, feed_dict={x_input: [test_x[step]]})
                predict = prob_output.reshape((-1))
                test_predict.append(predict[-1])
            test_predict = np.array(test_predict)*std+mean
            
            trigger = 0.0006 # 只有预测打败trigger，才会进行交易
            trading_fee = 0.0006
            
            signals =  [[0,0,0,0]]
            location = 0
            for i in range(1,len(test_predict)):
                if signals[location][3] == 0:
                    if (test_predict[i]-test_y[i-1])>=trigger:
                        re = [i-1,test_y[i-1],0,0]
                        re[2] = 1
                        re[3] = 1
                        signals.append(re)
                        location = location+1
                    elif (test_predict[i]-test_y[i-1])<=-trigger:
                        re = [i-1,test_y[i-1],0,0]
                        re[2] = 3
                        re[3] = 2
                        signals.append(re)
                        location = location+1
                elif signals[location][3] == 1:
                    if ((test_predict[i]-test_y[i-1])<0)&((test_predict[i]-test_y[i-1])>-trigger):
                        re = [i-1,test_y[i-1],0,0]
                        re[2] = 2
                        re[3] = 0
                        signals.append(re)
                        location = location+1
                    elif (test_predict[i]-test_y[i-1])<=-trigger:
                        re = [i-1,test_y[i-1],0,0]
                        re[2] = 6
                        re[3] = 2
                        signals.append(re)
                        location = location+1
                elif signals[location][3] == 2:
                    if ((test_predict[i]-test_y[i-1])>0)&((test_predict[i]-test_y[i-1])<trigger):
                        re = [i-1,test_y[i-1],0,0]
                        re[2] = 4
                        re[3] = 0
                        signals.append(re)
                        location = location+1
                    elif (test_predict[i]-test_y[i-1])>=trigger:
                        re = [i-1,test_y[i-1],0,0]
                        re[2] = 5
                        re[3] = 1
                        signals.append(re)
                        location = location+1
            if signals[location][3]==1:
                re = [i,test_y[i],2,0]
                signals.append(re)
                location = location+1
            elif signals[location][3]==2:
                re = [i,test_y[i],4,0]
                signals.append(re)
                location = location+1   
            
            for i in range(len(signals)):
                if signals[i][3] == 0:
                    signals[i].append(float('nan'))
                elif signals[i][3] == 1:
                    income = (test_y[signals[i+1][0]]-test_y[signals[i][0]])/test_y[signals[i][0]]-trading_fee/test_y[signals[i][0]]
                    signals[i].append(income)
                else:
                    income = (test_y[signals[i][0]]-test_y[signals[i+1][0]])/test_y[signals[i][0]]-trading_fee/test_y[signals[i][0]]
                    signals[i].append(income)
            del signals[0]
            my_signals.append([date,signals])
            print(date)
    return my_signals

