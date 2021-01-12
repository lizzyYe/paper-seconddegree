import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import datetime
import math
# import akshare as ak

token='ae6900908cb22edf8016c76c08552534f3779b07d03418c5a9470c7b'
ts.set_token('ae6900908cb22edf8016c76c08552534f3779b07d03418c5a9470c7b')

pro = ts.pro_api(token)

##################"""历史模拟法"""#########################
# SARS最初2002年十一月出现到2003年8月得到全面控制，2004年4月有复发
#计算sars疫情前7个月的收益波动率
data_p=ts.get_k_data('000001','2002-04-01','2002-12-01',autype='hfq')
data_p['d_return'] = data_p['close'].pct_change()
data_p['d_return']=[math.log(x+1) for x in data_p.d_return]

ini_o=data_p.d_return.std()

data_pre=ts.get_k_data('000001','2002-12-15','2003-07-13',autype='hfq')
data_pre['d_return'] = data_pre['close'].pct_change()
data_pre['d_return']=[math.log(x+1) for x in data_pre.d_return]
data_pre=data_pre[1:]

pre_data=np.array(data_pre[['date','d_return','close']])

#疫情开始前7个月
dd=ts.get_k_data('000001','2019-04-01','2019-12-01',autype='hfq')
dd['d_return'] = dd['close'].pct_change()
dd['d_return']=[math.log(x+1) for x in dd.d_return]

# 新冠肺炎最初于2019年十二月出现，直至现在
data=ts.get_k_data('000001','2019-11-28',str(datetime.date.today()),autype='hfq')#2019-12-1至今
data['d_return'] = data['close'].pct_change()
data['d_return']=[math.log(x+1) for x in data.d_return]
data=data[1:]

data_train=data[:int(round(data.shape[0]*0.7,0))]#前70%作为训练
data_test=data[int(round(data.shape[0]*0.7,0)):]#后30%作为测试

close_ini=data_train.iloc[0].close
print('sars_std:2002.4~2003.12',ini_o)
print('sars_std:2002.12~2003.7',data_pre.d_return.std())
print('std covid-19: 2019.4~2019.12',dd.d_return.std())
print('data__std covid-19: 2019.12.02~2020.6.15',data.d_return.std())
print('data_train_std covid-19: 2019.12.02~2020.4.16',data_train.d_return.std())
print('data_test_std covid-19 :2020.4.17.~2020.6.15',data_test.d_return.std())

train_data=np.array(data_train[['date','d_return','close']])[1:,:]
train_data=np.concatenate((pre_data,train_data),axis=0)
test_data=np.array(data_test[['date','d_return','close']])

def culo(arr,m,ini):
    arr=list(arr)
    result=[ini]
    for u in arr:
        result.append(math.sqrt(m*ini**2+(1-m)*u**2))
    return result[:-1]

# new_col = culo(train_data[:, 1],0.94, ini_o)
# print('天数','               波动率          ')
# print(np.array(list(enumerate(new_col))))

# 疫情现状


#计算情境下对数收益-一次
def cullnr(arr_v,o_now,arr_o):
    result=[]
    for i in range(1,len(arr_v)):
        result.append(math.log((arr_v[i-1]+(arr_v[i]-arr_v[i-1])*o_now/float(arr_o[i]))/float(arr_v[i-1])))
    return result

#计算情景下的值，储存为【日期，排序后收益率】
def culsit(train_data,ini_o,lam=0.94,nums=91):
    new_col = culo(train_data[:, 1],lam, ini_o)  # EWMA估计波动率
    results = []
    for i in range(nums):
        index_now = 132 + i
        arr_v = train_data[i:index_now, 2]
        o_now = new_col[index_now]
        arr_o = new_col[i:index_now]
        result0 = math.log((close_ini + (train_data[132,2] - close_ini) * o_now/ new_col[132]) / close_ini)
        result=cullnr(arr_v, o_now, arr_o)
        if i!=0:
            result[-i]=result0
        results.append([train_data[index_now, 0]] + sorted(result))
    return np.array(results)

def compare_sum(arr):
    # arr=list(arr)
    return sum([float(x)-float(y) for x,y in arr if float(x)>float(y)])/130 #收益为负是损失#损失平均大了多少


def cul_best_lambda(train_data,ini_o,lams=[x/100 for x in range(1,100,5)],nums=91):
    #0.01，0.06，0.11，...，0.96 20个
    sums=[]
    for lam in lams:
        results=culsit(train_data,ini_o,lam,nums)
        VaR_5=results[:,7] #131*0.05=6.55 取7
        VaR_1=results[:,2] #131*0.01=1.31 取2
        sums.append([compare_sum(np.concatenate((train_data[132:,1].reshape(nums,1),VaR_5.reshape(nums,1)),axis=1)),compare_sum(np.concatenate((train_data[132:,1].reshape(nums,1),VaR_1.reshape(nums,1)),axis=1))])
    sums=np.array(sums)
    plt.plot(lams,sums[:,0],label='VaR_5')
    plt.plot(lams,sums[:,1],label='VaR_1')
    plt.legend()
    plt.show()
    print('')

#对测试集进行评估，选取相对较优lambda 0.56,0.94
cul_best_lambda(train_data,ini_o)
lams=[0.56,0.94]
data_for_test=np.concatenate((train_data,test_data),axis=0)
cul_best_lambda(data_for_test,ini_o,lams,nums=130)
