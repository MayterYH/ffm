import numpy as np
import pandas as pd
from astropy.time import Time
from interval import Interval
import matplotlib.pyplot as plt


def LIS_1(e_in):
    beta = (e_in ** 2 + 2 * e_in * 0.938) ** 0.5 / (e_in + 0.938)
    j_lis = 2.7 * (10 ** 3) * (e_in ** 1.12) / beta ** 2 * ((e_in + 0.67) / 1.67) ** (-3.93)
    return j_lis


def LIS_2(e_in):
    beta = (e_in ** 2 + 2 * e_in * 0.938) ** 0.5 / (e_in + 0.938)
    j_lis = 3719 / (beta ** 2) * e_in ** 1.03 * ((e_in ** 1.21 + 0.77 ** 1.21) / (1 + 0.77 ** 1.21)) ** (-3.18)
    return j_lis


def LIS_3(e_in):
    beta = (e_in ** 2 + 2 * e_in * 0.938) ** 0.5 / (e_in + 0.938)
    j_lis = 2620 / (beta ** 2) * e_in ** 1.1 * ((e_in ** 0.98 + 0.7 ** 0.98) / (1 + 0.7 ** 0.98)) ** (-4) + \
            30 * e_in ** 2 * ((e_in + 8) / 9) ** (-12)
    return j_lis


def LIS_4(e_in):
    R = ((e_in + 0.938) ** 2 - 0.938 ** 2) ** 0.5
    beta = (e_in ** 2 + 2 * e_in * 0.938) ** 0.5 / (e_in + 0.938)
    # j_lis = 11740 * (1 + np.exp(-(np.log(R) + 0.559) / 0.563)) ** (-1 / 0.4315) * R ** (-2.4482) * \
    #         (1 + ((R / 6.2) * (1 + (R / 545) ** (0.6 / 0.4)) ** (-0.4)) ** (-0.4227 / (-0.108))) ** (-0.108)

    j_lis = 11740 * (1 + np.exp(-(np.log(R) + 0.559) / 0.563)) ** (-1 / 0.4315) * R ** (-2.4482) * \
            (1 + ((R / 6.2) * (1 + ((R / 545) ** (0.6 / 0.4))) ** (-0.4)) ** (0.4227 / 0.108)) ** (-0.108)
    return j_lis / beta


def LIS_5(e_in):
    j_lis = np.array([])
    for i in e_in:
        # print(i)
        if i <= 1.4:
            j_lis_0 = 707 * np.exp(4.64 - 0.036 * (np.log(i)) ** 2 - 2.91 * i ** 0.5)
            j_lis = np.append(j_lis, j_lis_0)
        elif i > 1.4:
            j_lis_1 = 685 * np.exp(3.22 - 2.78 * np.log(i) - 1.5 / i)
            j_lis = np.append(j_lis, j_lis_1)
    return j_lis


def LIS_6(e_in):
    beta = (e_in ** 2 + 2 * e_in * 0.938) ** 0.5 / (e_in + 0.938)
    R = ((e_in + 0.938) ** 2 - 0.938 ** 2) ** 0.5

    def LL(x):
        result = 1 / (1 + np.exp(-x))
        return result

    def RR(x):
        result = np.log(x)
        return result

    def GG(x):
        result = np.exp(-(x ** 2))
        return result

    j_lis = np.array([])
    for j in R:
        if j <= 2.5:
            j_lis_0 = 321.81 * j ** 2 + 11729 * LL(j) ** 2 + (2923.2 + 10621 * RR(j)) * GG(j / LL(j)) - \
                      1386.7 - 10399 * RR(j) - 0.67166 * LL(j) * 10528 ** (GG(j / LL(j)))
            j_lis = np.append(j_lis, j_lis_0)
        else:
            j_lis_1 = j ** (-2.7) * (
                    -2824 - 1.743 * 10 ** (-3) * j + 14742 * LL(j) + 2661.7 * GG(5.2830 * 10 ** (-2) * j) +
                    (171.60 * RR(j) - 19222 ** (LL(j)) * 0.15) * np.cos(0.9479 + 0.849 * RR(j))
            )
            j_lis = np.append(j_lis, j_lis_1)
    return j_lis / beta


def LIS_5_1(e_in):
    if e_in <= 1.4:
        j_lis = 707 * np.exp(4.64 - 0.036 * (np.log(e_in)) ** 2 - 2.91 * e_in ** 0.5)
    elif e_in > 1.4:
        j_lis = 685 * np.exp(3.22 - 2.78 * np.log(e_in) - 1.5 / e_in)
    return j_lis


def LIS_6_1(e_in):
    beta = (e_in ** 2 + 2 * e_in * 0.938) ** 0.5 / (e_in + 0.938)
    R = ((e_in + 0.938) ** 2 - 0.938 ** 2) ** 0.5

    def LL(x):
        result = 1 / (1 + np.exp(-x))
        return result

    def RR(x):
        result = np.log(x)
        return result

    def GG(x):
        result = np.exp(-(x ** 2))
        return result

    if R <= 2.5:
        j_lis = 321.81 * R ** 2 + 11729 * LL(R) ** 2 + (2923.2 + 10621 * RR(R)) * GG(R / LL(R)) - \
                1386.7 - 10399 * RR(R) - 0.67166 * LL(R) * 10528 ** (GG(R / LL(R)))
    else:
        j_lis = R ** (-2.7) * (
                -2824 - 1.743 * 10 ** (-3) * R + 14742 * LL(R) + 2661.7 * GG(5.2830 * 10 ** (-2) * R) +
                (171.60 * RR(R) - 19222 ** (LL(R)) * 0.15) * np.cos(0.9479 + 0.849 * RR(R))
        )
    return j_lis / beta


def LIS_7(e_in):
    c = [3.4617, -4.131, -4.6403, -1.4058, -4.7537, 8.5077, 32.637,
         -28.383, -58.203, 48.129, 33.946, -29.586, 0.61683]

    def log10J(x):
        result = 0
        for k in range(13):
            y_out = c[k] * (np.log10(x) / np.log10(800)) ** k
            result += y_out
        print(result)
        return np.power(10, result)

    return log10J(e_in)


def ffm_fun(e, phi):
    e_lis = e + phi
    # beta = (e_lis ** 2 + 2 * e_lis * 0.938) ** 0.5 / (e_lis + 0.938)
    j_lis = LIS_5_1(e_lis)
    result = j_lis * e * (e + 2 * 0.938) / (e + phi) / (e + phi + 2 * 0.938)

    return result


def ffm_fun_all(e, phi):
    e_lis = e + phi
    # beta = (e_lis ** 2 + 2 * e_lis * 0.938) ** 0.5 / (e_lis + 0.938)
    j_lis = LIS_5(e_lis)
    result = j_lis * e * (e + 2 * 0.938) / (e + phi) / (e + phi + 2 * 0.938)

    return result


def fit_obj_vary(E_input, phi_0, b):
    # import point_fit
    # z, E_0 = point_fit.run()
    a = -1

    # z = -1
    # # E_0 = 2
    # z = -0.8
    # E_0 = 2.5
    # c = -1
    # para = pd.read_csv(r'./data/output/ze.csv', header=0, index_col=0)
    # c = para.iloc[0, 0]
    # z = para.iloc[1, 0]
    # E_0 = para.iloc[2, 0]
    # c, z, E_0 = [-1.46665425, -1.1640105, 4.79466287]
    c, z, E_0 = [-1.2628246, -0.95064057, 3.91517157]
    c, z, E_0 = [-1.49095911, - 1.19474765, 4.87535719]
    # c, z, E_0 = [-1.87001184, -2., 7.5470661]
    beta = (E_input ** 2 + 2 * E_input * 0.938) ** 0.5 / (E_input + 0.938)
    Phi = phi_0 * beta ** a * E_input ** b * (1 + (E_input / E_0) ** (c / z)) ** z
    return Phi


# def fit_obj_stable(E_input, phi_0, b, c, z, E_0):
#     a = -1
#
#     # z = -1
#     # E_0 = 2
#     # z = -0.76
#     # E_0 = 3.23
#
#     beta = (E_input ** 2 + 2 * E_input * 1) ** 0.5 / (E_input + 1)
#     Phi = phi_0 * beta ** a * E_input ** b * (1 + (E_input / E_0) ** (c / z)) ** z
#     return Phi

# def reduce_dimension(data_in):
#     dimension = data_in.values.ndim
#
#     if dimension == 1 & len(data_in.values) == 79:
#         data = np.tile(data_in.values, (83, 1)).ravel()
#         return data
#     else:
#         data = data_in.values.ravel()
#         return data


def obj(x_in, a, b):
    return a * x_in + b


def ssn_phi(ssn):
    return 0.00481575 * ssn + 0.43897584


def ssn_b(ssn):
    return -0.00132168 * ssn + 0.43943567


def sigmoid(x):
    y = -2. / (1 + np.exp(-x / 3)) + 1
    return y


def delay_ssn(ssn, dealy):
    """
    :param ssn:太阳黑子数 序列 结束时刻与GC数据一致
    :param dealy: array_lisk 对应月份的延迟数值 负值
    :return: 延迟后的太阳黑子数
    """
    Delayssn = np.array([])

    for i, ssn_val in enumerate(ssn):
        if i == len(dealy):
            break
        ssn_d = ssn[-len(dealy) + i + dealy[i]]
        Delayssn = np.append(Delayssn, ssn_d)

    return Delayssn


def sigmod_smooth(gap):
    x_in = np.array(range(-int(gap), int(gap) + 1, 1))
    y_out = sigmoid(x_in)
    p1 = (y_out + 1) / 2
    p2 = (1 - y_out) / 2

    return p1, p2


def PtoN(ssn, delay_1, delay_2):
    """
    :param ssn:最后一个时期截止的太阳黑子数
    :param delay_1: 分界点前的延迟月数 array_like len = gap
    :param delay_2: 分界点后的延迟月数 array_like len = gap
    :return: 平滑后的 太阳黑子数
    """
    x_in = np.array(range(-12, 13, 1))
    y_out = sigmoid(x_in)
    p1 = (y_out + 1) / 2
    p2 = (1 - y_out) / 2
    # print(p1)
    ssn_1 = delay_ssn(ssn, delay_1)
    ssn_2 = delay_ssn(ssn, delay_2)
    ssn_out = p1 * ssn_1 + p2 * ssn_2

    return ssn_out


def fun_ssn(x, a, b, c, d):
    y = a + b * x * np.log(x) + c * (np.log(x)) ** 2 + d * (np.log(x)) ** 0.5
    return y


def t_jd(t_in):
    return Time(t_in, format='iso').jd1


def fun_line(y1, y2, det_x):
    return ((y2 - y1) / det_x) * np.linspace(0, det_x, det_x + 1) + y1


def fun_delay(time):
    while time in Interval(t_jd('1966-8-1'), t_jd('1972-11-1'), upper_closed=False):
        time_diff = int(t_jd('1972-11-1') - t_jd('1966-8-1'))
        dealy = fun_line(6, 0, time_diff)
        return - int(dealy[int(time - t_jd('1972-11-1'))])

    while time in Interval(t_jd('1972-11-1'), t_jd('1976-5-1'), upper_closed=False):
        return 0

    while time in Interval(t_jd('1976-5-1'), t_jd('1978-5-1'), upper_closed=False):
        return -6

    while time in Interval(t_jd('1978-5-1'), t_jd('1983-3-1'), upper_closed=False):
        time_diff = int(t_jd('1983-3-1') - t_jd('1978-5-1'))
        dealy = fun_line(6, 18, time_diff)
        return - int(dealy[int(time - t_jd('1978-5-1'))])

    while time in Interval(t_jd('1983-3-1'), t_jd('1986-5-1'), upper_closed=False):
        time_diff = int(t_jd('1986-5-1') - t_jd('1983-3-1'))
        dealy = fun_line(18, 6, time_diff)
        return - int(dealy[int(time - t_jd('1983-3-1'))])

    while time in Interval(t_jd('1986-5-1'), t_jd('1988-4-1'), upper_closed=False):
        return -6
    # =============================================================================
    while time in Interval(t_jd('1988-4-1'), t_jd('1993-1-1'), upper_closed=False):
        time_diff = int(t_jd('1993-1-1') - t_jd('1988-4-1'))
        dealy = fun_line(6, 0, time_diff)
        return - int(dealy[int(time - t_jd('1988-4-1'))])

    while time in Interval(t_jd('1993-1-1'), t_jd('1996-6-1'), upper_closed=False):
        return 0

    while time in Interval(t_jd('1996-6-1'), t_jd('1998-7-1'), upper_closed=False):
        return -6

    while time in Interval(t_jd('1998-7-1'), t_jd('2003-11-1'), upper_closed=False):
        time_diff = int(t_jd('2003-11-1') - t_jd('1998-7-1'))
        dealy = fun_line(6, 18, time_diff)
        return - int(dealy[int(time - t_jd('1998-7-1'))])

    while time in Interval(t_jd('2003-11-1'), t_jd('2008-11-1'), upper_closed=False):
        time_diff = int(t_jd('2008-11-1') - t_jd('2003-11-1'))
        dealy = fun_line(18, 6, time_diff)
        return - int(dealy[int(time - t_jd('2003-11-1'))])

    while time in Interval(t_jd('2008-11-1'), t_jd('2011-11-1'), upper_closed=False):
        return -6
    # ===============================================================================
    while time in Interval(t_jd('2011-11-1'), t_jd('2016-2-1'), upper_closed=False):
        time_diff = int(t_jd('2016-2-1') - t_jd('2011-11-1'))
        dealy = fun_line(6, 0, time_diff)
        return - int(dealy[int(time - t_jd('2011-11-1'))])

    while time in Interval(t_jd('2016-2-1'), t_jd('2020-1-1'), upper_closed=False):
        return 0


def LR_negative(x_in):
    """
    :param x_in: 二维数组 第一列 ssn 第二列 alpha
    :return: 一维数组 计算出的 phi_0
    """
    # return 0.0285566 * x_in[0] + 0.00705183 * x_in[1] + 0.22449633  # ln ssn l_av
    # return 0.03585935 * x_in[0] + 0.00556126 * x_in[1] + 0.34299927  # ln ssn r_av
    # return 0.01966609 * x_in[0] + 0.1941944 * x_in[1] - 0.0615641 # log
    # return 0.00386936 * x_in[0] + 0.00098895 * x_in[1] + 0.44581549 # line
    # return 0.05379475 * x_in[0] + 0.11878411 * x_in[1] + 0.06980006  # log-13
    # return 0.12291513 * x_in[0] + 0.00125525 * x_in[1] + 0.2341584  # m-l-

    para = pd.read_csv(r'../test_LR_3/output/LR_para.csv', header=0, index_col=0)
    para_n = para.iloc[0, :].values
    # print(para_n)
    return para_n[0] * x_in[0] + para_n[1] * x_in[1] + para_n[2]


def LR_postive(x_in):
    """
    :param x_in: 二维数组 第一列 ssn 第二列 alpha
    :return: 一维数组 计算出的 phi_0
    """
    # return 0.15798539 * x_in[0] + 0.01118619 * x_in[1] - 0.5713798  # ln ssn l_av
    # return 0.18940888 * x_in[0] + 0.00631102 * x_in[1] - 0.34948979  # ln ssn r_av
    # return 0.11131437 * x_in[0] + 0.38017057 * x_in[1] - 1.16647202 # log
    # return 0.0036249 * x_in[0] + 0.00472219 * x_in[1] + 0.25703738 # line
    # return 0.2039945 * x_in[0] + 0.25242544 * x_in[1] - 1.05478436  # log-13
    # return 0.04491461 * x_in[0] + 0.18914814 * x_in[1] - 0.12257033  # log-13-plus
    # return 0.14681811 * x_in[0] + 0.0058427 * x_in[1] - 0.11732801  # log-13-plus-log-line
    # return 0.16992066 * x_in[0] - 0.00569928 * x_in[1] + 0.05258269  # m-l
    para = pd.read_csv(r'../test_LR_3/output/LR_para.csv', header=0, index_col=0)
    para_p = para.iloc[1, :].values
    # print(para_p)
    # print(para_p)
    return para_p[0] * x_in[0] + para_p[1] * x_in[1] + para_p[2]


def LR_all(x_in):
    para = pd.read_csv(r'../test_LR_3/output/LR_para_all.csv', header=0, index_col=0)
    para_a = para.iloc[:, 0].values
    return para_a[0] * x_in[0] + para_a[1] * x_in[1] + para_a[2] * x_in[2] + para_a[3]


def fun_alpha(x):
    if x < 0.24:
        y = 5 + 1131.1 * x ** 2
    elif (x <= 0.32) & (x >= 0.24):
        y = 70
    elif x > 0.32:
        y = 5 + 141.1 * (1 - x) ** 2
    return y
