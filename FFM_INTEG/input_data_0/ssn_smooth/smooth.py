import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import time_sequence

"""
平滑太阳黑子数 去掉双峰结构
"""

df_ssn = pd.read_csv(r'./ssn_smooth_plus_test.csv', header=0, index_col=0)
print(df_ssn)
ssn = df_ssn.iloc[:, 0].values
ssn_smooth = savgol_filter(ssn, 51, 2, mode='nearest')
ssnTimelist = list(time_sequence.generate('1970-1-1', '2026-1-1').index)
time_list = [ssnTimelist[i][:-3] for i in range(len(ssnTimelist))]

plt.figure(figsize=(9, 4), dpi=200)
plt.plot(range(len(ssn)), ssn)
plt.plot(range(len(ssn_smooth)), ssn_smooth)
plt.show()
print(ssn_smooth)
"""
保存数据到文件 已注释
"""
# pd.DataFrame(np.vstack((ssn, ssn_smooth)).T,
#              index=time_list, columns=['ssn', 'smooth']).to_csv(r'./ssn_smooth_plus_test.csv')
