# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import time
import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from scipy.optimize import rosen, differential_evolution
from scipy.optimize import basinhopping
from scipy.optimize import brute
from scipy.optimize import shgo

from ffm_Phi_1 import DATA
import FUNC

nn = 0  # 拟合的时期

phi_pam, err_pam = DATA.pamela_all()
e_pam = phi_pam.columns.values

phi_ams, err_ams = DATA.ams02_all()
e_ams = phi_ams.columns.values

cp_pam = 61
cp_ams = 23

e_pam = e_pam[0:cp_pam]
phi_pam = phi_pam.iloc[:, 0:cp_pam].values
err_pam = err_pam.iloc[:, 0:cp_pam].values

e_ams = e_ams[0:cp_ams]
phi_ams = phi_ams.iloc[:, 0:cp_ams].values
err_ams = err_ams.iloc[:, 0:cp_ams].values


# %%

def fit_obj(E_input, phi_in, b_in, c_in, z_in, E_0):
    a = -1
    beta = (E_input ** 2 + 2 * E_input * 1) ** 0.5 / (E_input + 1)
    Phi = phi_in * beta ** a * E_input ** b_in * (1 + (E_input / E_0) ** (c_in / z_in)) ** z_in
    return Phi


def kapa(p_in, e_in, phi_in):
    c = p_in[0]
    z = p_in[1]
    E_0 = p_in[2]

    def fit_obj_vary(E_input, phi_0, b):
        a = -1
        beta = (E_input ** 2 + 2 * E_input * 1) ** 0.5 / (E_input + 1)
        Phi = phi_0 * beta ** a * E_input ** b * (1 + (E_input / E_0) ** (c / z)) ** z
        return Phi

    param_bounds = ([0, 0],
                    [3, 2])  # 0.42
    result = np.array([])
    # for i in range(83):
    print('-' * 50)
    print(p_in)
    for i in range(81):
        # print(p_in)
        popt, pcov = curve_fit(fit_obj_vary, e_in[i * cp_pam:(i + 1) * cp_pam], phi_in[i * cp_pam:(i + 1) * cp_pam],
                               sigma=err_hd[i * cp_pam:(i + 1) * cp_pam], absolute_sigma=True,
                               bounds=param_bounds,
                               maxfev=100000)

        pcov = np.sqrt(np.diag(pcov))
        result_sigle = ((fit_obj(e_in[i * cp_pam:(i + 1) * cp_pam], popt[0], popt[1], c, z, E_0)
                         - phi_in[i * cp_pam:(i + 1) * cp_pam]) / err_hd[i * cp_pam:(i + 1) * cp_pam]) ** 2

        # print(i)
        result = np.append(result, result_sigle)

        # print(popt)
    print(result.sum())
    return result


e_hd_pam = np.tile(e_pam, (81, 1)).ravel()
print(e_hd_pam)
e_hd_ams = np.tile(e_ams, (68, 1)).ravel()

e_hd = np.append(e_hd_pam, e_hd_ams)


phi_hd_pam = phi_pam.ravel()
phi_hd_ams = phi_ams.ravel()
phi_hd = np.append(phi_hd_pam, phi_hd_ams)
print(len(phi_hd_pam))
print(len(phi_hd_ams))

err_hd_pam = err_pam.ravel()
err_hd_ams = err_ams.ravel()
err_hd = np.append(err_hd_pam, err_hd_ams)

print(phi_hd_pam[-1])
print(e_hd[len(phi_hd_pam):])
print(e_hd[5022 + 0 * cp_ams: 5022 + (0 + 1) * cp_ams])
print(len(phi_hd[5022 + 0 * cp_ams: 5022 + (0 + 1) * cp_ams]))


def kapa_0(p_in):
    c = p_in[0]
    z = p_in[1]
    E_0 = p_in[2]

    def fit_obj_vary(E_input, phi_0, b):
        a = -1
        beta = (E_input ** 2 + 2 * E_input * 1) ** 0.5 / (E_input + 1)
        Phi = phi_0 * beta ** a * E_input ** b * (1 + (E_input / E_0) ** (c / z)) ** z
        return Phi

    param_bounds = ([0, 0],
                    [3, 2])  # 0.42
    result = np.array([])
    # for i in range(83):
    print('-' * 35)
    print(p_in)
    for i in range(149):
        if i < 81:
            # print(p_in)
            popt, pcov = curve_fit(fit_obj_vary, e_hd[i * cp_pam:(i + 1) * cp_pam], phi_hd[i * cp_pam:(i + 1) * cp_pam],
                                   sigma=err_hd[i * cp_pam:(i + 1) * cp_pam], absolute_sigma=True,
                                   bounds=param_bounds,
                                   maxfev=100000)

            pcov = np.sqrt(np.diag(pcov))
            result_sigle = (((fit_obj(e_hd[i * cp_pam:(i + 1) * cp_pam], popt[0], popt[1], c, z, E_0)
                              - phi_hd[i * cp_pam:(i + 1) * cp_pam]) / err_hd[i * cp_pam:(i + 1) * cp_pam]) ** 2) ** 0.5

            # print(i)
            result = np.append(result, result_sigle)
            # print(result)
        elif (i >= 81) & (i <= 148):

            i -= 81

            cut_num = 81*cp_pam

            popt, pcov = curve_fit(fit_obj_vary, e_hd[cut_num + i * cp_ams:cut_num + (i + 1) * cp_ams],
                                   phi_hd[cut_num + i * cp_ams: cut_num + (i + 1) * cp_ams],
                                   sigma=err_hd[cut_num + i * cp_ams:cut_num + (i + 1) * cp_ams],
                                   absolute_sigma=True, bounds=param_bounds, maxfev=100000)

            pcov = np.sqrt(np.diag(pcov))
            result_sigle = (((fit_obj(e_hd[cut_num + i * cp_ams:cut_num + (i + 1) * cp_ams], popt[0],
                                      popt[1], c, z, E_0) - phi_hd[cut_num + i * cp_ams:cut_num + (i + 1) * cp_ams])
                             / err_hd[cut_num + i * cp_ams:cut_num + (i + 1) * cp_ams]) ** 2) ** 0.5

            result = np.append(result, result_sigle)

        # print(popt)
    # print(len(result))
    print(result.sum())
    return result.sum()


start = time.time()

# p = np.array([-1, -1, 3])


#
# res = minimize(kapa_0, p, method='BFGS')
# print(res.x, res.fun)

#
# para = leastsq(kapa, p, args=(e_hd, phi_hd), maxfev=100000)
#
# print('last time')
# print(para)

# # #
lw = [-2, -2, 1]
up = [0, 0, 10]
bounds = list(zip(lw, up))
# rres = dual_annealing(kapa_0, bounds=list(zip(lw, up)), seed=1234)
# print(rres.x, rres.fun)
# # [-1.5        -1.26037143  4.87486079]
# [-1.44917782 -1.14075328  4.70590667] 1429.7833546390693

# #
# result_0 = shgo(kapa_0, bounds=list(zip(lw, up)))
# print(result_0.x, result_0.fun)
# print(result_0)
# [-1.65019214 -1.45775513  5.66536954]
#
result_f = differential_evolution(kapa_0, bounds)
print(result_f.x, result_f.fun)
# [-1.65017985 -1.45773884  5.66530278]
# [-1.49234324 -1.19505346  4.91452854] 1429.7825508400115
# [-1.46665425 -1.1640105   4.79466287] 1429.744289090954
#
#
# class MyBounds(object):
#     def __init__(self, xmax=[0, 0, 10], xmin=[-1.5, -1.5, 1]):
#         self.xmax = np.array(xmax)
#         self.xmin = np.array(xmin)
#
#     def __call__(self, **kwargs):
#         x = kwargs["x_new"]
#         tmax = bool(np.all(x <= self.xmax))
#         tmin = bool(np.all(x >= self.xmin))
#         return tmax and tmin
#
#
# minimizer_kwargs = {"method": "BFGS"}
# x0 = [-1, -1, 3]
# mybounds = MyBounds()
# ret = basinhopping(kapa_0, x0, minimizer_kwargs=minimizer_kwargs, accept_test=mybounds,
#                    niter=200)
# print(ret.x)
#
#
# rranges = (slice(-2, 0, 0.01), slice(-2, 0, 0.01),
#            slice(0,10,0.01))
#
# resbrute = brute(kapa_0, rranges, full_output=True,
#                           finish=optimize.fmin)


end = time.time()
print('Running time: %s Seconds' % (end - start))

# [-1.25197267, -0.93816487,  3.86804913]  pamela
# [-1.20936917, -1.07904038,  2.76192438]  ams02
# [-0.92149749, -0.58761696,  2.56548064]  all
