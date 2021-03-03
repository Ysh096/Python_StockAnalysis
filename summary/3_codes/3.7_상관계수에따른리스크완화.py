# 3.7.1 데이터프레임으로 상관게수 구하기
# from pandas_datareader import data as pdr
# import pandas as pd
# import yfinance as yf
# yf.pdr_override()
#
# dow = pdr.get_data_yahoo('^DJI', '2000-01-04') # 다우존수 지수의 심볼 = ^DJI
# kospi = pdr.get_data_yahoo('^KS11', '2000-01-04') # 코스피
#
# # 데이터 프레임으로 두 시리즈 묶기
# df = pd.DataFrame({'DOW': dow['Close'], 'KOSPI': kospi['Close']})
#
# # NaN 부분 없애주기
# df = df.fillna(method='bfill')
# df = df.fillna(method='ffill')
#
# print(df.corr())


# 3.7.2 시리즈로 상관계수 구하기
# from pandas_datareader import data as pdr
# import pandas as pd
# import yfinance as yf
# yf.pdr_override()
#
# dow = pdr.get_data_yahoo('^DJI', '2000-01-04') # 다우존수 지수의 심볼 = ^DJI
# kospi = pdr.get_data_yahoo('^KS11', '2000-01-04') # 코스피
#
# # 데이터 프레임으로 두 시리즈 묶기
# df = pd.DataFrame({'DOW': dow['Close'], 'KOSPI': kospi['Close']})
#
# # NaN 부분 없애주기
# df = df.fillna(method='bfill')
# df = df.fillna(method='ffill')
#
# print(df['DOW'].corr(df['KOSPI']))
#
# # 결정계수
# r_squared = df['DOW'].corr(df['KOSPI']) ** 2
# print(r_squared)


# 3.7.4 다우존스 지수와 KOSPI의 회귀 분석
# from pandas_datareader import data as pdr
# import pandas as pd
# from scipy import stats
# import yfinance as yf
# import matplotlib.pyplot as plt
# yf.pdr_override()
#
# # 다우존스, 코스피 지수 데이터 가져오기
# dow = pdr.get_data_yahoo('^DJI', '2000-01-04') # 다우존수 지수의 심볼 = ^DJI
# kospi = pdr.get_data_yahoo('^KS11', '2000-01-04') # 코스피
#
# # 데이터 프레임으로 묶어주기
# df = pd.DataFrame({'DOW': dow['Close'], 'KOSPI': kospi['Close']})
# df = df.fillna(method='bfill')
# df = df.fillna(method='ffill')
#
# # 선형회귀식 도출
# regr = stats.linregress(df['DOW'], df['KOSPI'])
# regr_line = 'Y = {:.2f} * X + {:.2f}'.format(regr.slope, regr.intercept)
# print(regr_line)
#
# # 그려보기
# plt.figure(figsize=(7, 7))
# plt.scatter(df['DOW'], df['KOSPI'], marker='.')
# plt.plot(df['DOW'], regr.slope*df['DOW'] + regr.intercept, 'r')
# plt.legend(['DOW x KOSPI', regr_line])
# plt.title('DOW x KOSPI (R = {:.2f})'.format(regr.rvalue))
# plt.xlabel('Dow Jones Industrial Average')
# plt.ylabel('KOSPI')
# plt.show()


# 미국 국채와 KOSPI의 회귀 분석
from pandas_datareader import data as pdr
import pandas as pd
from scipy import stats
import yfinance as yf
import matplotlib.pyplot as plt
yf.pdr_override()

# 미국국채, 코스피 지수 데이터 가져오기
tlt = pdr.get_data_yahoo('TLT', '2000-01-04') # 미국국채 심볼 TLT
kospi = pdr.get_data_yahoo('^KS11', '2000-01-04') # 코스피

# 데이터 프레임으로 묶어주기
df = pd.DataFrame({'TLT': tlt['Close'], 'KOSPI': kospi['Close']})
df = df.fillna(method='bfill')
df = df.fillna(method='ffill')

# 선형회귀식 도출
regr = stats.linregress(df['TLT'], df['KOSPI'])
regr_line = 'Y = {:.2f} * X + {:.2f}'.format(regr.slope, regr.intercept)
print(regr_line)

# 그려보기
plt.figure(figsize=(7, 7))
plt.scatter(df['TLT'], df['KOSPI'], marker='.')
plt.plot(df['TLT'], regr.slope*df['TLT'] + regr.intercept, 'r')
plt.legend(['TLT x KOSPI', regr_line])
plt.title('TLT x KOSPI (R = {:.2f})'.format(regr.rvalue))
plt.xlabel('iShares Barclays 20 + Yr Treas.Bond(TLT)')
plt.ylabel('KOSPI')
plt.show()