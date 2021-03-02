from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

dow = pdr.get_data_yahoo('^DJI', '2000-01-04') # 다우존수 지수의 심볼 = ^DJI
kospi = pdr.get_data_yahoo('^KS11', '2000-01-04') # 코스피
# 1. 단순 비교
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9, 5))
# plt.plot(dow.index, dow.Close, 'r--', label='Dow Jones Industrial')
# plt.plot(kospi.index, kospi.Close, 'b', label = 'KOSPI')
# plt.grid(True)
# plt.legend(loc='best')
# plt.show()

# 2. 지수화 비교
# d = (dow.Close / dow.Close.loc['2000-01-04']) * 100
# k = (kospi.Close / kospi.Close.loc['2000-01-04']) * 100
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9, 5))
# plt.plot(dow.index, d, 'r--', label='Dow Jones Industrial Average')
# plt.plot(kospi.index, k, 'b', label='KOSPI')
# plt.grid(True)
# plt.legend(loc='best')
# plt.show()

# 3. 산점도
# import pandas as pd
# df = pd.DataFrame({'DOW': dow['Close'], 'KOSPI': kospi['Close']})
# df = df.fillna(method='bfill')
# df = df.fillna(method='ffill')
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7, 7))
# plt.scatter(df['DOW'], df['KOSPI'], marker='.')
# plt.xlabel('Dow Jones Industrial Average')
# plt.ylabel('KOSPI')
# plt.show()

from scipy import stats
import pandas as pd
df = pd.DataFrame({'DOW': dow['Close'], 'KOSPI': kospi['Close']})
df = df.fillna(method='bfill')
df = df.fillna(method='ffill')
regr = stats.linregress(df['DOW'], df['KOSPI'])
print(regr)