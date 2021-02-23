import os
import numpy as np
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
import time_sequence as ts
import FUNC

"""
读取PAMELA每个时期能量为 275MeV 的 通量保存到 pam_275.csv 加上儒略日信息
"""

df = pd.read_csv(r'../input_data_0/imp8_274_month.txt', header=None, index_col=None, delim_whitespace=True)
pam_list = os.listdir(r'../input_data_0/pam_mon/')
pam_275 = np.array([])
for i, name in enumerate(pam_list):
    df = pd.read_csv(r'../input_data_0/pam_mon/%s' % name, header=0, index_col=0)
    flux = df.iloc[13, 0]
    pam_275 = np.append(pam_275, flux)
index = [pam_list[i][-14:-4] for i in range(len(pam_list))]
jd_index = Time(index, format='iso').jd1 + 0.5
pd.DataFrame(np.vstack((jd_index, pam_275)).T, index=index).to_csv(r'./output/pam_275.csv')

"""
计算 SSN 的延迟信息 并且 计算出 phi_c ,b_c ,Phi_c, flux_c
"""

range_info = ts.generate('1974-1-1', '2020-12-1')
range_info['mark'] = np.ones(len(range_info), dtype='int') * -5
range_info.loc['1980-01-01':'1989-12-01', 'mark'] = \
    np.ones(len(range_info.loc['1980-01-01':'1989-12-01', 'mark']), dtype='int') * -12
range_info.loc['2000-06-01':'2012-06-01', 'mark'] = \
    np.ones(len(range_info.loc['2000-06-01':'2012-06-01', 'mark']), dtype='int') * -12
# # 按照 周期来分组
# range_info['mark'] = np.ones(len(range_info), dtype='int') * -6
# range_info.loc['1976-06-01':'1986-09-01', 'mark'] = \
#     np.ones(len(range_info.loc['1976-06-01':'1986-09-01', 'mark']), dtype='int') * -12
# range_info.loc['1996-05-01':'2008-12-01', 'mark'] = \
#     np.ones(len(range_info.loc['1996-05-01':'2008-12-01', 'mark']), dtype='int') * -12


# range_info.loc['1990-01-01':'1990-12-01', 'mark'] = [i for i in range(-12, 0, 1)]
# range_info.loc['2014-02-01':'2015-01-01', 'mark'] = [i for i in range(-12, 0, 1)]

# pass

"""
写入太阳黑子数
"""
df_ssn = pd.read_csv(r'../input_data_0/ssn_smooth_plus_test.csv', header=0, index_col=0)
ssn = df_ssn.loc[:'2020-12', 'smooth'].values
dealy = range_info.loc[:, 'mark'].values

delay_ssn = np.array([])
for i, ssn_val in enumerate(ssn):
    if i == len(dealy):
        break
    ssn_d = ssn[-len(dealy) + i + dealy[i]]
    delay_ssn = np.append(delay_ssn, ssn_d)

dealy_pos = np.ones(25, dtype='int') * -5
dealy_neg = np.ones(25, dtype='int') * -12

ssn_1 = df_ssn.loc[:'1980-12', 'smooth'].values
ssn_2 = df_ssn.loc[:'1990-12', 'smooth'].values
ssn_3 = df_ssn.loc[:'2001-05', 'smooth'].values
ssn_4 = df_ssn.loc[:'2013-06', 'smooth'].values

ssn_sig_1 = FUNC.PtoN(ssn_1, dealy_pos, dealy_neg)
ssn_sig_2 = FUNC.PtoN(ssn_2, dealy_neg, dealy_pos)
ssn_sig_3 = FUNC.PtoN(ssn_3, dealy_pos, dealy_neg)
ssn_sig_4 = FUNC.PtoN(ssn_4, dealy_neg, dealy_pos)

range_info['ssn_delay'] = delay_ssn
"""
sigmod 平滑
"""

range_info.loc['1978-12-01':'1980-12-01', 'ssn_delay'] = ssn_sig_1
range_info.loc['1988-12-01':'1990-12-01', 'ssn_delay'] = ssn_sig_2
range_info.loc['1999-05-01':'2001-05-01', 'ssn_delay'] = ssn_sig_3
range_info.loc['2011-06-01':'2013-06-01', 'ssn_delay'] = ssn_sig_4

# range_info.loc['1979-01-01':'1980-12-01', 'ssn_delay'] = \
#     np.linspace(range_info.loc['1979-01-01', 'ssn_delay'], range_info.loc['1980-12-01', 'ssn_delay'], 24)
#
# range_info.loc['1999-06-01':'2001-05-01', 'ssn_delay'] = \
#     np.linspace(range_info.loc['1999-06-01', 'ssn_delay'], range_info.loc['2001-05-01', 'ssn_delay'], 24)

'''
计算 phi_c ,b_c ,Phi_c, flux_c
'''
ssn_in = range_info.loc[:, 'ssn_delay'].values

plt.figure(figsize=(8,4))
plt.plot(range(len(delay_ssn)), delay_ssn, label='Data', c='b', lw=2,zorder=12)
plt.plot(range(len(delay_ssn)), ssn_in, label='Smooth', c='r', lw=2)
plt.legend()
plt.show()

phi_c = FUNC.ssn_phi(ssn_in)
b_c = FUNC.ssn_b(ssn_in)
Phi_c = FUNC.fit_obj_vary(0.274, phi_c, b_c)

flux_c = np.array([])
for i in range(len(Phi_c)):
    flux_sig = FUNC.ffm_fun(0.274, Phi_c[i])
    flux_c = np.append(flux_c, flux_sig)
# print(range_info)
range_info['phi_c'] = phi_c
range_info['b_c'] = b_c
range_info['Phi_c'] = Phi_c
range_info['flux_c'] = flux_c

range_info.to_csv(r'./output/info_test.csv')
