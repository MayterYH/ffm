import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import FUNC

"""
利用FFM拟合出单个能量点的PHI
输入：每个月的 能量，通量
输出：每个月的调制势PHI
"""


class PAMELA:

    def __init__(self, path):
        self.path = path

    def get_phi(self):
        df = pd.read_csv(self.path, header=0, index_col=0)
        # periods = int(self.path[-7:-4])
        np.seterr(invalid='ignore')
        e_data = df.index.values
        flux = df.iloc[:, 0].values
        flux_err = df.iloc[:, 1].values
        phi_period = np.array([])
        err_period = np.array([])

        for i in np.arange(len(e_data)):
            sigma_val = np.array([flux_err[i], ])
            para, pcov = curve_fit(FUNC.ffm_fun, e_data[i], flux[i],
                                   sigma=sigma_val, absolute_sigma=True)
            phi = para
            err = np.sqrt(np.diag(pcov))

            phi_period = np.append(phi_period, phi)
            err_period = np.append(err_period, err)
        e_period = np.array(e_data)
        return e_period, phi_period, err_period


#
class AMS02:

    def __init__(self, path):
        self.path = path

    def get_phi(self):
        df = pd.read_csv(self.path, header=0, index_col=0)
        # periods = int(self.path[-7:-4])
        np.seterr(invalid='ignore')
        e_data = df.index.values
        flux = df.iloc[:, 0].values
        flux_err = df.iloc[:, 1].values

        phi_period = np.array([])
        err_period = np.array([])

        for i in np.arange(len(e_data)):
            sigma_val = np.array([flux_err[i], ])
            para, pcov = curve_fit(FUNC.ffm_fun, e_data[i], flux[i],
                                   sigma=sigma_val, absolute_sigma=True)
            phi = para
            err = np.sqrt(np.diag(pcov))

            phi_period = np.append(phi_period, phi)
            err_period = np.append(err_period, err)
        e_period = np.array(e_data)
        return e_period, phi_period, err_period


def pamela_all():
    phi_all = np.array([])
    err_all = np.array([])

    path_all = r'../input_data_0/pam_mon/'
    for i in os.listdir(path_all):
        path = path_all + i
        e_in, phi_in, err_in = PAMELA(path).get_phi()

        phi_all = np.append(phi_all, phi_in, axis=0)
        err_all = np.append(err_all, err_in, axis=0)

    data_list = os.listdir(r'../input_data_0/pam_mon/')

    df_phi_all = pd.DataFrame(phi_all.reshape(81, 78), columns=e_in,
                              index=[data_list[j][-14:-7] for j in range(len(data_list))])
    df_err_all = pd.DataFrame(err_all.reshape(81, 78), columns=e_in,
                              index=[data_list[j][-14:-7] for j in range(len(data_list))])
    return df_phi_all, df_err_all


#
#
def ams02_all():
    phi_all = np.array([])
    err_all = np.array([])

    path_all = r'../input_data_0/ams_mon/'
    for m in os.walk(path_all):
        for n in sorted(m[2]):
            path = path_all + n
            e_in, phi_in, err_in = AMS02(path).get_phi()

            phi_all = np.append(phi_all, phi_in, axis=0)
            err_all = np.append(err_all, err_in, axis=0)

    data_list = os.listdir(r'../input_data_0/ams_mon/')

    df_phi_all = pd.DataFrame(phi_all.reshape(68, 45), columns=e_in,
                              index=[data_list[j][-14:-7] for j in range(len(data_list))])
    df_err_all = pd.DataFrame(err_all.reshape(68, 45), columns=e_in,
                              index=[data_list[j][-14:-7] for j in range(len(data_list))])
    return df_phi_all, df_err_all
