from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt

kospi = pdr.get_data_yahoo('^KS11', '2004-01-04') # Kospi 지수의 심볼 = ^KS11

window = 252 # 1년 개장일 어림값
peak = kospi['Close'].rolling(window, min_periods=1).max()
drawdown = kospi['Close']/peak - 1.0 # peak 대비 얼마나 하락했는가? 매일 달라짐
print(drawdown)
max_dd = drawdown.rolling(window, min_periods=1).min() #drawdown의 252일 중 최저치(window는 계속 움직임)
print(max_dd)
plt.figure(figsize=(9, 7))
plt.subplot(211) # 2행 1열 중 1행에 그림
kospi['Close'].plot(label='KOSPI', title='KOSPI MDD', grid=True, legend = True)
plt.subplot(212) # 2행 1 열 중 2행에 그림
drawdown.plot(c='blue', label='KOSPI DD', grid=True, legend = True)
max_dd.plot(c='red', label='KOSPI MDD', grid=True, legend=True)
plt.show()