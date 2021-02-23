import pandas as pd
import numpy as np

"""
利用线性插值 补充缺失月份的 phi_0 b
为下一步计算相关性系数做准备
"""

pam = pd.read_csv(r'./output/pamela_vary_2.csv', header=0, index_col=0)
ams = pd.read_csv(r'./output/ams02_vary_2.csv', header=0, index_col=0)

pam_err = pd.read_csv(r'./output/pamela_err_2.csv', header=0, index_col=0)
ams_err = pd.read_csv(r'./output/ams02_err_2.csv', header=0, index_col=0)

pam_nan = pd.read_csv(r'./output/pamela_mon_nan.csv', header=0, index_col=0)
ams_nan = pd.read_csv(r'./output/ams02_mon_nan.csv', header=0, index_col=0)

nan_pam = [int(i) for i in pam_nan.iloc[:, 0]]
nan_ams = [int(i) for i in ams_nan.iloc[:, 0]]

pam_phi = pam.iloc[:, 0].values
pam_b = pam.iloc[:, 1].values

pamPhierr = pam_err.iloc[:, 0].values
pamBerr = pam_err.iloc[:, 1].values

ams_phi = ams.iloc[:, 0].values
ams_b = ams.iloc[:, 1].values

amsPhierr = ams_err.iloc[:, 0].values
amsBerr = ams_err.iloc[:, 1].values


# ===============================================================
# PAMELA
def fill_pam():
    count = 0
    phi_all = np.array([])
    b_all = np.array([])
    phi_all_err = np.array([])
    b_all_err = np.array([])
    for i in range(90):
        if i in nan_pam:
            count += 1
            phi_all = np.append(phi_all, np.nan)
            b_all = np.append(b_all, np.nan)
            phi_all_err = np.append(phi_all_err, np.nan)
            b_all_err = np.append(b_all_err, np.nan)
        else:
            phi_all = np.append(phi_all, pam_phi[i - count])
            b_all = np.append(b_all, pam_b[i - count])
            phi_all_err = np.append(phi_all_err, pamPhierr[i - count])
            b_all_err = np.append(b_all_err, pamBerr[i - count])

    phi_all = pd.Series(phi_all).interpolate(method='linear')
    b_all = pd.Series(b_all).interpolate(method='linear')

    phi_all_err = pd.Series(phi_all_err).interpolate(method='linear')
    b_all_err = pd.Series(b_all_err).interpolate(method='linear')
    # print(phi_all.values)
    # print(len(phi_all.values))
    pd.DataFrame(np.vstack((phi_all, b_all)).T).to_csv(r'./output/fill_all/pamela_all.csv')
    pd.DataFrame(np.vstack((phi_all_err, b_all_err)).T).to_csv(r'./output/fill_all/pamela_all_err.csv')


# ===============================================================
# AMS02
def fill_ams():
    count = 0
    phi_all = np.array([])
    b_all = np.array([])
    phi_all_err = np.array([])
    b_all_err = np.array([])
    for i in range(70):
        if i in nan_ams:
            count += 1
            phi_all = np.append(phi_all, np.nan)
            b_all = np.append(b_all, np.nan)
            phi_all_err = np.append(phi_all_err, np.nan)
            b_all_err = np.append(b_all_err, np.nan)
        else:
            phi_all = np.append(phi_all, ams_phi[i - count])
            b_all = np.append(b_all, ams_b[i - count])
            phi_all_err = np.append(phi_all_err, amsPhierr[i - count])
            b_all_err = np.append(b_all_err, amsBerr[i - count])

    phi_all = pd.Series(phi_all).interpolate(method='linear')
    b_all = pd.Series(b_all).interpolate(method='linear')

    phi_all_err = pd.Series(phi_all_err).interpolate(method='linear')
    b_all_err = pd.Series(b_all_err).interpolate(method='linear')
    # print(phi_all.values)
    # print(len(phi_all.values))
    pd.DataFrame(np.vstack((phi_all, b_all)).T).to_csv(r'./output/fill_all/ams02_all.csv')
    pd.DataFrame(np.vstack((phi_all_err, b_all_err)).T).to_csv(r'./output/fill_all/ams02_all_err.csv')


if __name__ == "__main__":
    fill_pam()
    fill_ams()
