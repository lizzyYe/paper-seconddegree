# 获取至今的股票数据

import tushare as ts
import datetime



class get_data:
    def __init__(self,index_code,start_time,file_path,end_time=str(datetime.date.today())):
        token = 'ae6900908cb22edf8016c76c08552534f3779b07d03418c5a9470c7b'
        ts.set_token('ae6900908cb22edf8016c76c08552534f3779b07d03418c5a9470c7b')

        pro = ts.pro_api(token)
        data = ts.get_k_data(str(index_code), str(start_time), str(end_time), autype='hfq')
        try:
            data.to_csv(file_path)
            print('文件已保存至'+file_path)
        except:
            print('文件保存失败')



