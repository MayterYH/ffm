import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import FUNC
import matplotlib.pyplot as plt

"""
根据前面的到的数据 计算出ssn与phi，b的关系
"""

# 读取 phi_0 b 以及延迟时间 太阳黑子数的信息
df_pam = pd.read_csv(r'../double_two_2/output/fill_all/pamela_all.csv', header=0, index_col=0)
df_ams = pd.read_csv(r'../double_two_2/output/fill_all/ams02_all.csv', header=0, index_col=0)

df_pam_err = pd.read_csv(r'../double_two_2/output/fill_all/pamela_all_err.csv', header=0, index_col=0)
df_ams_err = pd.read_csv(r'../double_two_2/output/fill_all/ams02_all_err.csv', header=0, index_col=0)

info_pam = pd.read_csv(r'./input/pamela_info.csv', header=0, index_col=0)
info_ams = pd.read_csv(r'./input/ams02_info.csv', header=0, index_col=0)

# 为了处理方便 将延迟信息 太阳黑子数 加载到 phi — b 文件
df_pam.columns = ['phi_0', 'b']
df_pam['mark'] = info_pam.loc[:, 'lu_mark'].values
df_pam['ssn_del'] = info_pam.loc[:, 'lu_delay'].values

df_ams.columns = ['phi_0', 'b']
df_ams['mark'] = info_ams.loc[:, 'lu_mark'].values
df_ams['ssn_del'] = info_ams.loc[:, 'lu_delay'].values

x_ssn = np.append(df_pam.iloc[:, 3].values, df_ams.iloc[:, 3].values)
y_phi = np.append(df_pam.iloc[:, 0].values, df_ams.iloc[:, 0].values)
y_b = np.append(df_pam.iloc[:, 1].values, df_ams.iloc[:, 1].values)
y_phi_err = np.append(df_pam_err.iloc[:, 0].values, df_ams_err.iloc[:, 0].values)
y_b_err = np.append(df_pam_err.iloc[:, 1].values, df_ams_err.iloc[:, 1].values)

# 计算 ssn-phi/b 的系数
phi_para, _ = curve_fit(FUNC.obj, x_ssn, y_phi, sigma=y_phi_err, absolute_sigma=True)
b_para, _ = curve_fit(FUNC.obj, x_ssn, y_b, sigma=y_b_err, absolute_sigma=True)
print('phi:', phi_para)
print('b:', b_para)

# 讲关系式写入FFM 后计算 phi_c ，b_c 并且加载到 phi-b 文件

df_pam['phi_c'] = FUNC.ssn_phi(df_pam.loc[:, 'ssn_del'].values)
df_pam['b_c'] = FUNC.ssn_b(df_pam.loc[:, 'ssn_del'].values)

df_ams['phi_c'] = FUNC.ssn_phi(df_ams.loc[:, 'ssn_del'].values)
df_ams['b_c'] = FUNC.ssn_b(df_ams.loc[:, 'ssn_del'].values)
# 删除空缺的时间 并保存到 output 的 info文件
pam_nan = pd.read_csv(r'../double_two_2/output/pamela_mon_nan.csv', header=0, index_col=0)
ams_nan = pd.read_csv(r'../double_two_2/output/ams02_mon_nan.csv', header=0, index_col=0)

pam_nan_list = [int(pam_nan.iloc[:, 0][i]) for i in range(len(pam_nan))]
ams_nan_list = [int(ams_nan.iloc[:, 0][i]) for i in range(len(ams_nan))]

df_pam = df_pam.drop(index=pam_nan_list)
df_ams = df_ams.drop(index=ams_nan_list)

# 加上 index 保存到 output
pam_index = os.listdir(r'../input_data_0/pam_mon/')
ams_index = os.listdir(r'../input_data_0/ams_mon')

df_pam.index = [pam_index[i][-14:-7] for i in range(len(pam_index))]
df_ams.index = [ams_index[i][-14:-7] for i in range(len(ams_index))]

df_pam.to_csv(r'./output/pam_info.csv')
df_ams.to_csv(r'./output/ams_info.csv')

print(df_pam)
print(df_ams)

