# import datetime
# # import math
# # # print(str(datetime.date.today()- datetime.timedelta(days=1)))
# #
# list1=[1,2]
# list2=[3]
# print(list1+list2)
# print(math.log(math.e))


import akshare as ak
#
# CN = ak.covid_19_163(indicator="中国历史累计数据")
# CN_NEW = ak.covid_19_163(indicator="中国各地区时点数据")
# # CN.to_csv("累计.csv",sep=',',index=True,header=True)
# print('')

import tushare as ts
import datetime
import math
# token='ae6900908cb22edf8016c76c08552534f3779b07d03418c5a9470c7b'
# ts.set_token('ae6900908cb22edf8016c76c08552534f3779b07d03418c5a9470c7b')

# pro = ts.pro_api()
# data=ts.get_k_data('000001','2010-01-01',str(datetime.date.today()),autype='hfq')
# data=data[["close","open","high","low","volume"]]
# data=data.apply(lambda x:(x-min(x))/(max(x)-min(x)))
# print(data[0:5])
# sents=[]
# import numpy
# data=data.values.tolist()
# sents.append(data[0:5])
# print(sents)

# data=ts.pro_bar(ts_code='000001.SH',adj='hfq',start_date='20170120',end_date='20200731')
# data=ts.get_h_data('000001','2017-01-20',str(datetime.date.today()))#2019-12-1至今
# data['d_return'] = data['close'].pct_change()
# data['d_return']=[math.log(x+1) for x in data.d_return]
# data=data[1:]
# data.to_csv("shareprice_sh.csv",sep=',',index=False,header=True)
# print('')
#
# seq_len=4
# LOG_FILE = 'final_{0}.log'.format(seq_len)
# print(LOG_FILE)
import matplotlib.pyplot as plt
seq=[1,2,3,4]
plt.plot(range(len(seq)),seq)
plt.show()