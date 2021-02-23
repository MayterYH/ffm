import numpy as np


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
    else:
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
    c, z, E_0 = [-1.46665425, -1.1640105, 4.79466287]
    # c, z, E_0 = [-1.2628246, -0.95064057, 3.91517157]
    # c, z, E_0 = [-1.87001184, -2., 7.5470661]
    c, z, E_0 = [-1.49095911, - 1.19474765, 4.87535719]
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
