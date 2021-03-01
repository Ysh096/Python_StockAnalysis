# 인덱스와 컬럼이 어떻게 생겼는가?
# from pandas_datareader import data as pdr
# sec = pdr.get_data_yahoo('005930.KS', start = '2020-03-01')
# aapl = pdr.get_data_yahoo('AAPL', start = '2020-03-01')
# print(sec.index)
# print(sec.columns)

# 단순 주가 비교
# from pandas_datareader import data as pdr
# import matplotlib.pyplot as plt
# sec = pdr.get_data_yahoo('005930.KS', start = '2020-03-01')
# aapl = pdr.get_data_yahoo('AAPL', start = '2020-03-01')
#
# plt.plot(sec.index, sec.Close, 'b', label='Samsung Electronics') # 푸른 색 실선
# plt.plot(aapl.index, aapl.Close, 'r--', label='Apple') # 빨간색 -- 선
# plt.legend(loc='best') # location = best, 그림이 겹치지 않는 가장 좋은 위치에 범례를 표시해준다.
# plt.show()

# 일간 변동률 표현 방법
# sec_dpc = (sec['Close'] / sec['Close'].shift(1) - 1) * 100
# print(sec_dpc.head())


# from pandas_datareader import data as pdr
# import matplotlib.pyplot as plt
# sec = pdr.get_data_yahoo('005930.KS', start = '2020-03-01')
# aapl = pdr.get_data_yahoo('AAPL', start = '2020-03-01')
# sec_dpc = (sec['Close']-sec['Close'].shift(1)) / sec['Close'].shift(1) * 100
# sec_dpc.iloc[0] = 0
# plt.hist(sec_dpc, bins=18) #18개 구간으로 나누기
# plt.grid(True)
# plt.show()

# # 일간 변동률 누적합
# from pandas_datareader import data as pdr
# sec = pdr.get_data_yahoo('005930.KS', start = '2020-03-01')
# aapl = pdr.get_data_yahoo('AAPL', start = '2020-03-01')
# sec_dpc = (sec['Close']-sec['Close'].shift(1)) / sec['Close'].shift(1) * 100
# aapl_dpc = (aapl['Close']-aapl['Close'].shift(1)) / aapl['Close'].shift(1) * 100
# sec_dpc.iloc[0] = 0
# aapl_dpc.iloc[0] = 0
# sec_dpc_cs = sec_dpc.cumsum()
# aapl_dpc_cs = aapl_dpc.cumsum()
# print(sec_dpc_cs)
# print(aapl_dpc_cs)

# 일간 변동률 누적합 그래프 그려보기
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

# 야후 API로 주가 정보 가져오기
sec = pdr.get_data_yahoo('005930.KS', start = '2020-03-01')
aapl = pdr.get_data_yahoo('AAPL', start = '2020-03-01')

# 일일 주가 변동률 계산
sec_dpc = (sec['Close']-sec['Close'].shift(1)) / sec['Close'].shift(1) * 100
aapl_dpc = (aapl['Close']-aapl['Close'].shift(1)) / aapl['Close'].shift(1) * 100

# 첫 행의 데이터를 NaN이 아닌 0으로 바꿔줌
sec_dpc.iloc[0] = 0
aapl_dpc.iloc[0] = 0

# 누적 주가 변동률 계산
sec_dpc_cs = sec_dpc.cumsum()
aapl_dpc_cs = aapl_dpc.cumsum()

plt.plot(sec.index, sec_dpc_cs, 'b', label='Samsung Electronics') # x축은 인덱스, y축은 누적 주가 변동률
plt.plot(aapl.index, aapl_dpc_cs, 'r--', label='Apple') # 즉 날짜별 수익률을 표시함!
plt.ylabel('Change %')
plt.grid(True) # 그리드 선 그리기(바둑판)
plt.legend(loc='best') # 가장 적절한 위치에 범례 표시
plt.show()