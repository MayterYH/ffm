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

phi, err = DATA.pamela_all()
e = phi.columns.values

cp = 69

e = e[0:cp]

print(e)
phi = phi.iloc[:, 0:cp].values
err = err.iloc[:, 0:cp].values


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
    for i in range(83):
        # print(p_in)
        popt, pcov = curve_fit(fit_obj_vary, e_in[i * cp:(i + 1) * cp], phi_in[i * cp:(i + 1) * cp],
                               sigma=err_hd[i * cp:(i + 1) * cp], absolute_sigma=True,
                               bounds=param_bounds,
                               maxfev=100000)

        pcov = np.sqrt(np.diag(pcov))
        result_sigle = ((fit_obj(e_in[i * cp:(i + 1) * cp], popt[0], popt[1], c, z, E_0)
                         - phi_in[i * cp:(i + 1) * cp]) / err_hd[i * cp:(i + 1) * cp]) ** 2

        # print(i)
        result = np.append(result, result_sigle)

        # print(popt)
    print(result.sum())
    return result


e_hd = np.tile(e, (83, 1)).ravel()
phi_hd = phi.ravel()
err_hd = err.ravel()


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
    for i in range(81):
        # print(p_in)
        popt, pcov = curve_fit(fit_obj_vary, e_hd[i * cp:(i + 1) * cp], phi_hd[i * cp:(i + 1) * cp],
                               sigma=err_hd[i * cp:(i + 1) * cp], absolute_sigma=True,
                               bounds=param_bounds,
                               maxfev=100000)

        pcov = np.sqrt(np.diag(pcov))
        result_sigle = (((fit_obj(e_hd[i * cp:(i + 1) * cp], popt[0], popt[1], c, z, E_0)
                          - phi_hd[i * cp:(i + 1) * cp]) / err_hd[i * cp:(i + 1) * cp]) ** 2) ** 0.5

        # print(i)
        result = np.append(result, result_sigle)

        # print(popt)
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
