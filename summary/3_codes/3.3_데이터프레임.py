import pandas as pd
# df = pd.DataFrame({'KOSPI': [1915, 1961, 2026, 2467, 2041],
#                    'KOSDAQ': [542, 682, 631, 798, 675]},
#                   index = [2014, 2015, 2016, 2017, 2018])
# # print(df)
# # print(df.describe())
# print(df.info())

# 시리즈 이용
# kospi = pd.Series([1915, 1961, 2026, 2467, 2041],
#                   index = [2014, 2015, 2016, 2017, 2018], name='KOSPI')
#
# kosdaq = pd.Series([542, 682, 631, 798, 675],
#                    index=[2014, 2015, 2016, 2017, 2018], name='KOSDAQ')
#
# df = pd.DataFrame({kospi.name: kospi, kosdaq.name: kosdaq})
# print(df)

#리스트 이용
columns = ['KOSPI', 'KOSDAQ']
index = [2014, 2015, 2016, 2017, 2018]
rows = []
rows.append([1915, 542])
rows.append([1961, 682])
rows.append([2026, 631])
rows.append([2467, 798])
rows.append([2041, 675])
df = pd.DataFrame(rows, columns=columns, index=index)
print(rows)
print(df)