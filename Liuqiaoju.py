import numpy as np
import pandas as pd
import tushare as ts
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import MySQLdb as mdb

import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection.univariate_selection import f_classif

#获取股票代码
con = mdb.connect(host = 'localhost',user = 'root',passwd = '123456',
                db = 'lqj',use_unicode = True,charset = 'utf8')
sql = 'select ts_code as tsymbol,name as names from h_stock'
data = pd.read_sql(sql,con)

stock1 = ','.join(data[data['names'] == '贵州茅台']['tsymbol'].values)
stock2 = ','.join(data[data['names'] == '平安银行']['tsymbol'].values)
print(stock1,stock2)


#获取stock1,stock2每日价格
pro = ts.pro_api()
stocks = [stock1,stock2]
price_stocks=[]
for symbol in stocks:
    price = pro.daily(ts_code = symbol,start_date = '20150102',
                        end_date = '20181201')
    price = price.set_index('trade_date')
    price = price[::-1].drop_duplicates()
    price.index = pd.to_datetime(price.index,format='%Y%m%d')
    price_stocks.append(price)

print('Sample size of stock1 is:{}'.format(len(price_stocks[0])))
print('Sample size of stock2 is:{}'.format(len(price_stocks[1])))

    
##########################################################################
#计算回报值R，R=当日收盘价-前一日的收盘价#######################################
##########################################################################

return_stocks=[]
for price in price_stocks:
    p = price['close']
    p_1 = price['close'].shift(1)
    result=(p-p_1)/p_1
    return_stocks.append(result)

    
return_cumsum = []
return_sum = []
return_describe = []
return_skew = []
return_kurt = []
for i in range(2):
    return_cumsum.append(return_stocks[i].cumsum())
    return_sum.append(return_stocks[i].sum())
    return_describe.append(return_stocks[i].describe())
    return_skew.append(return_stocks[i].skew())
    return_kurt.append(return_stocks[i].kurt())
        

##########################################################################
###########共用变量#########################################################
##########################################################################
    
stock_name = ['MouTai','PingAn Bank']

##########################################################################
###########第一问的图#######################################################
########################################################################## 
#直方图
for i in range(2):
    plt.subplot(1,2,i+1)
    n, bins, patches = plt.hist(return_stocks[0],48,
                                density=True,facecolor='white',
                                edgecolor='black',
                                alpha=0.5)
    mu = round(return_stocks[i].mean(),3)
    sigma = round(return_stocks[i].std(),3)
    maxv =round(return_stocks[i].max(),3)
    minv =round(return_stocks[i].min(),3)
    y=stats.norm.pdf(bins,mu,sigma)
    plt.plot(bins,y,'r--',linewidth=0.6)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.annotate('Max&Min:{},{}'.format(maxv,minv),
            xy=(0.04,17),fontsize=10,color='blue')
    ax.annotate(r'$\mu$&$\delta$:{},{}'.format(sigma,mu),
            xy=(0.04,16),fontsize=10,color='blue')
    ax.text(0.03,13,'The red dashed line',
            fontsize=9,color='r',style='italic')
    ax.text(0.03,12,'is normal distribution',
            fontsize=9,color='r',style='italic')
    plt.xlabel('Return of {}'.format(stock_name[i]))
    plt.title('Figure{} Histogram of return'.format(i+1))
    if i == 0:
        plt.ylabel('Density/%')
    else:
        continue
plt.show()

#盒须图
plt.figure(figsize=(8,6),dpi=80,facecolor=(0.96,0.97,0.94))
for i in range(2):
    a = return_stocks[i]
    a.name = stock_name[i]
    plt.subplot(1,2,i+1)
    a.plot.box()
    if i == 0:
        plt.ylabel('Return')
    else:
        continue
plt.suptitle('Figure3 Boxplot of return from two stocks')
plt.show()



##########################################################################
###########第二问的检验：18年波动幅度是否比17年大#######################################
##########################################################################
#重新取样
sample = []
abs_return1 = np.abs(return_stocks[0])

sample.append(abs_return1.loc['2017-01-03':'2017-12-31'])
sample.append(abs_return1.loc['2018-01-03':'2018-12-31'])

#shapiro检验
from scipy.stats import shapiro
shapiro_p = []
for i in range (2):
    p=shapiro(np.log(sample[i].replace(0,np.nan)))
    shapiro_p.append(p)
    
#求F值
sigma12 = (sample[0].std())**2
sigma22 = (sample[1].std())**2

f = max(sigma12/sigma22,sigma22/sigma12)
f = round(f,3)
#F检验，并根据根据检验结果进行T检验。
cpf = stats.distributions.f.cdf(f,len(sample[0])-1,len(sample[1])-1)
p = 1-cpf
p = round(p,3)
if p < 0.025:
    equal_var = False
    print('Variances are equal')
else:
    equal_var = True
    print('Variances are not equal')
test_result = stats.ttest_ind(sample[0],sample[1],equal_var = equal_var)
t_p = round(test_result.pvalue,3)
if t_p < 0.025:
    print('The mean between two smaple_17 and 18 is equal')
else:
    print('The mean between two smaple_17 and 18 is not equal')



##########################################################################
###########第二问检验和图###################################################
##########################################################################
#求样本均值和方差
s0 = round(sample[0].std(),3)
mu0 = round(sample[0].mean(),3)
s1 = round(sample[1].std(),3)
mu1 = round(sample[1].mean(),3)

#作图
plt.figure(figsize=(8,6),dpi=80,facecolor=(0.96,0.97,0.94))
for i in range(2):
    plt.subplot(2,2,i+1)
    plt.hist(sample[i],bins = 48,facecolor='white',edgecolor='black')
    plt.xlabel('About value of return of {}'.format(2017+i))
    ax = plt.gca()  # 获得当前axis,'get current axis'
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    sigma = sample[i].std()
    if i == 0:
        plt.ylabel('Count')
    else:
        continue
    
for i in range(3,5):
    plt.subplot(2,2,i)
    if i == 3:
        plt.text(0.1,0.6,'The result of f test:',fontsize = 10)
        plt.text(0.2,0.5,'f={}; pvalue={}'.format(f,p),fontsize = 10)
        plt.text(0.1,0.4,'The result of t test:',fontsize = 10)
        plt.text(0.2,0.3,'pvalue={}'.format(t_p),fontsize = 10)
    else:
        plt.text(0.2,0.6,r'2018:$\mu$={}, $\delta$={}'.format(s1,mu1),
        fontsize=10)
        plt.text(0.2,0.4,r'2017:$\mu$={}, $\delta$={}'.format(s0,mu0),
                 fontsize=10)
        
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
plt.suptitle('Figure4 Histogram of absulote values of return(MouTai)')
plt.show()


##########################################################################
###########第三问的解答：做简单线性回归###################################################
##########################################################################
return_value = return_stocks[1]
return_fvalue = return_value.shift(1)
regr_data = pd.DataFrame(index = return_value.index)
regr_data['return_value'] = return_value
regr_data['return_fvalue'] = return_fvalue
regr_data = regr_data.dropna()
X = regr_data['return_fvalue']
y = regr_data['return_value']
model = sm.OLS(y,X)
res_model = model.fit()
total_params=res_model.summary()

plt.scatter(X,y,s=20,marker='o')
plt.xlabel('Previous trading day return')
plt.ylabel('A trading day return')
plt.title('Figure5 Scatter plot of return in a given trading day against the previos trading day',fontsize=14)
plt.show()  
