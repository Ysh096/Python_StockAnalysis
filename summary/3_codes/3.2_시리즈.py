import pandas as pd

# 시리즈 생성
s = pd.Series([0.0, 3.6, 2.0, 5.8, 4.2, 8.0])
s.index = pd.Index([0.0, 1.2, 1.8, 3.0, 3.6, 4.8])
s.index.name = 'MY_IDX'

# 시리즈 정보 보기
s.describe()

# 시리즈 출력하기
s = pd.Series([0.0, 3.6, 2.0, 5.8, 4.2, 8.0, 5.5, 6.7, 4.2])
s.index = pd.Index([0.0, 1.2, 1.8, 3.0, 3.6, 4.8, 5.9, 6.8, 8.0])

s.index.name = 'MY_DIX'
s.name = 'MY_SERIES'

import matplotlib.pyplot as plt
plt.title("ELLIOTT_WAVE")
plt.plot(s, 'bs--')
plt.show()
plt.xticks(s.index)
plt.yticks(s.values)

