#获取实时分笔数据
import threading
import time
import tushare as ts

code='000581'
def data(code):
    df=ts.get_realtime_quotes(code)
    print(df.head())
i=0
while i<3:
    print(time.time())
    t=threading.Timer(4,data,[code])
    t.start()
    t.join()
    i += 1

