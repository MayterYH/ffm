import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os

from ffm_Phi_1 import DATA

"""
 利用双幂律方程拟合出 phi_0 b
 输入：ffm_Phi_1 的到的单个能量点的Phi
 输出：双幂指律方程的参数 phi_0 b (每个月一组)
"""


def run():
    def fit_obj_1(E_input, phi_0, b):
        a = -1
        c, z, E_0 = [-1.2628246, -0.95064057, 3.91517157]  # differential_evolution
        beta = (E_input ** 2 + 2 * E_input * 0.938) ** 0.5 / (E_input + 0.938)
        Phi = phi_0 * beta ** a * E_input ** b * (1 + (E_input / E_0) ** (c / z)) ** z
        return Phi

    # ========================pamela=============================================
    df_phi, err = DATA.pamela_all()
    e = df_phi.columns.values

    cp = 61

    e = e[0:cp]
    phi = df_phi.iloc[:, 0:cp].values
    err = err.iloc[:, 0:cp].values

    para = np.array([])
    para_err = np.array([])

    for i in range(81):
        param_bounds = ([0, 0],
                        [3, 2])  # 0.42

        popt, pocv = curve_fit(fit_obj_1, e, phi[i],
                               sigma=err[i], absolute_sigma=True,
                               bounds=param_bounds,
                               maxfev=100000)

        pcov = np.sqrt(np.diag(pocv))
        print('*****PAMELA第%d个时期*****' % i)
        print(popt)

        para = np.append(para, popt, axis=0)
        para_err = np.append(para_err, pcov, axis=0)

    para = pd.DataFrame(para.reshape(81, 2), index=df_phi.index, columns=['phi_0', 'b'])
    para.to_csv(r'./output/pamela_vary_2.csv')
    para_err = pd.DataFrame(para_err.reshape(81, 2), index=df_phi.index, columns=['phi_0', 'b'])
    para_err.to_csv(r'./output/pamela_err_2.csv')

    # ===========================================================================

    df_phi, err = DATA.ams02_all()
    e = df_phi.columns.values

    cp = 23

    e = e[0:cp]
    phi = df_phi.iloc[:, 0:cp].values
    err = err.iloc[:, 0:cp].values

    para = np.array([])
    para_err = np.array([])
    for i in range(68):
        param_bounds = ([0, 0],
                        [3, 2])  # 0.42

        popt, pocv = curve_fit(fit_obj_1, e, phi[i],
                               sigma=err[i], absolute_sigma=True,
                               bounds=param_bounds,
                               maxfev=100000)

        pcov = np.sqrt(np.diag(pocv))
        print('*****AMS02第%d个时期*****' % i)
        print(popt)
        print(pcov)
        para = np.append(para, popt, axis=0)
        para_err = np.append(para_err, pcov, axis=0)

    parameter = pd.DataFrame(para.reshape(68, 2), index=df_phi.index, columns=['phi_0', 'b'])
    parameter.to_csv(r'./output/ams02_vary_2.csv')
    para_err = pd.DataFrame(para_err.reshape(68, 2), index=df_phi.index, columns=['phi_0', 'b'])
    para_err.to_csv(r'./output/ams02_err_2.csv')


if __name__ == "__main__":
    run()
