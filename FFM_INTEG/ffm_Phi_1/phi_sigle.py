import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import FUNC
from ffm_Phi_1 import DATA

"""
 存储 利用FFM方程拟合出的不随能量变化的 PHI
"""

pam, pam_err = DATA.pamela_all()
ams, ams_err = DATA.ams02_all()
pam.to_csv(r'./output/pam_Phi.csv')
ams.to_csv(r'./output/ams_Phi.csv')
pam_err.to_csv(r'./output/pam_Phi_err.csv')
ams_err.to_csv(r'./output/ams_Phi_err.csv')


def sigle_phi():
    file = os.listdir(r'../input_data_0/pam_mon/')
    pam_phi = np.array([])
    for i, name in enumerate(file):
        df = pd.read_csv(r'../input_data_0/pam_mon/%s' % name, header=0, index_col=0)
        e = df.index.values
        flux = df.iloc[:, 0].values
        error = df.iloc[:, 1].values
        para, pocv = curve_fit(FUNC.ffm_fun_all, e, flux, sigma=error, absolute_sigma=True)
        pam_phi = np.append(pam_phi, para[0])

    file = os.listdir(r'../input_data_0/ams_mon/')
    ams_phi = np.array([])
    for j, name in enumerate(file):
        df = pd.read_csv(r'../input_data_0/ams_mon/%s' % name, header=0, index_col=0)
        e = df.index.values
        flux = df.iloc[:, 0].values
        error = df.iloc[:, 1].values
        para, pocv = curve_fit(FUNC.ffm_fun_all, e, flux, sigma=error, absolute_sigma=True)
        ams_phi = np.append(ams_phi, para[0])

    print(len(pam_phi))
    print(len(ams_phi))

    pam_index = os.listdir(r'../input_data_0/pam_mon/')
    ams_index = os.listdir(r'../input_data_0/ams_mon')

    pd.DataFrame(pam_phi, index=[pam_index[i][-14:-7] for i in range(len(pam_index))]) \
        .to_csv(r'./output/pam_phi_sig.csv')

    pd.DataFrame(ams_phi, index=[ams_index[i][-14:-7] for i in range(len(ams_index))]) \
        .to_csv(r'./output/ams_phi_sig.csv')


if __name__ == "__main__":
    sigle_phi()
