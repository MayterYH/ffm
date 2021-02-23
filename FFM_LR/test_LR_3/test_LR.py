import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

import FUNC

"""
先将alpha数据做延迟 然后整合到info文件 为线性回归做准备
"""

alpha_type = 'r_smooth'

pam = pd.read_csv(r'../input_data_0/pamela_all.csv', header=0, index_col=0)
ams = pd.read_csv(r'../input_data_0/ams02_all.csv', header=0, index_col=0)
df_alpha = pd.read_csv(r'../mon_alpha_1/output/alpha_mon.csv', header=0, index_col=0)

fix_ssn = 0
fix_log_ssn = 0
fix_alpha = 0

# pam_ssn = df_pam.loc[:, 'ssn'].values + fix_ssn
# ams_ssn = df_ams.loc[:, 'ssn'].values + fix_ssn

df_pam_ssn = pd.read_csv(r'../ssn_flux_2/output/info_test.csv',
                         header=0, index_col=0)
pam_ssn = df_pam_ssn.loc['2006-08-01':'2014-01-01', 'ssn_delay'].values + fix_ssn
print(df_pam_ssn.loc['2006-08-01':'2014-01-01', 'ssn_delay'])
pam_phi = pam.iloc[:, 0].values

df_ams_ssn = pd.read_csv(r'../ssn_flux_2/output/info_test.csv',
                         header=0, index_col=0)
ams_ssn = df_ams_ssn.loc['2011-06-01':'2017-03-01', 'ssn_delay'].values + fix_ssn
ams_phi = ams.iloc[:, 0].values

pam_log_ssn = pam_ssn ** 0.5 + fix_log_ssn
ams_log_ssn = ams_ssn ** 0.5 + fix_log_ssn
# print(pam_ssn)

pam_end = df_alpha.loc[:'2014-01', alpha_type]
pam_dealy = np.ones(len(pam_ssn), dtype='int') * -12
pam_alpha = FUNC.delay_ssn(pam_end, pam_dealy) + fix_alpha

ams_end = df_alpha.loc[:'2017-03', alpha_type]
ams_dealy = np.ones(len(ams_ssn), dtype='int') * -12
ams_alpha = FUNC.delay_ssn(ams_end, ams_dealy) + fix_alpha
print(ams_alpha)

# pam_alpha = np.log(pam_alpha)
# ams_alpha = np.log(ams_alpha)

df_pam = pd.DataFrame(np.vstack((pam_phi, pam_ssn, pam_log_ssn, pam_alpha)).T,
                      columns=['phi_0', 'ssn', 'log_ssn_fix', alpha_type],
                      index=df_pam_ssn.loc['2006-08-01':'2014-01-01', :].index)
df_ams = pd.DataFrame(np.vstack((ams_phi, ams_ssn, ams_log_ssn, ams_alpha)).T,
                      columns=['phi_0', 'ssn', 'log_ssn_fix', alpha_type],
                      index=df_ams_ssn.loc['2011-06-01':'2017-03-01', :].index)
print(df_pam)

"""
A
"""
# print(list(df_pam.index))
pam_x = (Time(list(df_pam.index), format='iso').jd1 - Time('2012-06-01', format='iso').jd1) / 30
ams_x = (Time(list(df_ams.index), format='iso').jd1 - Time('2012-06-01', format='iso').jd1) / 30
pam_A = np.array([format(-FUNC.sigmoid(pam_x)[i], '.3f') for i in range(len(pam_x))], dtype='float')
ams_A = np.array([format(-FUNC.sigmoid(ams_x)[i], '.3f') for i in range(len(ams_x))], dtype='float')
df_pam['A'] = pam_A
df_ams['A'] = ams_A
# df_pam.to_csv(r'./test.csv')
# df_ams.plot(y='A')
# plt.show()
print(df_pam)
print(df_ams)
# df_pam[alpha_type] = pam_alpha
# df_ams[alpha_type] = ams_alpha
#
# df_pam.to_csv(r'./output/pam_info.csv')
# df_ams.to_csv(r'./output/ams_info.csv')
#
# df_pam['ssn_fix'] = pam_ssn
# df_ams['ssn_fix'] = ams_ssn
# df_pam['log_ssn_fix'] = pam_log_ssn
# df_ams['log_ssn_fix'] = ams_log_ssn
# plt.plot(range(len(df_pam)))

"""
以A的正负为界限做线性回归 以 ssn 和 alpha 为自变量 phi 为因变量 
"""

df_all = pd.read_csv(r'../input_data_0/all_fit_info.csv', header=0, index_col=0)


# df_ams[[alpha_type]] = (df_ams.loc[:,alpha_type]) ** 0.5
# df_pam[[alpha_type]] = (df_pam.loc[:,alpha_type]) ** 0.5


def LR_negative():
    # time_ssn = pd.read_csv(r'./output/time_ssn.csv', header=0, index_col=0)
    # pam_ssn = time_ssn.loc['2006-08':'2014-01', 'log_ssn']
    # ams_ssn = time_ssn.loc['2011-06':'2017-03', 'log_ssn']
    """
    对于A小于0的情况
    """
    df_all.loc[:, 'log_ssn'] = np.log(df_all[['ssn']].values + fix_ssn)
    # print(df_all.loc[:, 'log_ssn'])
    all_neg_para = df_all.iloc[-5:-3, [3, 5]].values
    # print(len(all_neg_para))
    # print(all_neg_para)
    #
    # print(len(pam_ssn))
    # print(len(df_pam.loc[:, 'log_ssn_fix'].values))
    # df_pam['log_ssn'] = pam_ssn.values
    # df_ams['log_ssn'] = ams_ssn.values
    # print(df_pam)
    x_pam = df_pam.loc[:'2012-06-01', ['log_ssn_fix', alpha_type]].values
    x_ams = df_ams.loc[:'2012-06-01', ['log_ssn_fix', alpha_type]].values
    #  df_ams.loc[:'2008-12', ['log_ssn_fix', alpha_type]].values
    x = np.vstack((x_pam, x_ams, all_neg_para))

    # po = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    # x_poly = po.fit_transform(x)
    # df_nolin = pd.DataFrame(x_poly)
    # print(df_nolin)

    # scaler = StandardScaler().fit(x)
    # # print(scaler.transform(x))
    # x = scaler.transform(x)
    # print(x)
    all_nega_phi = df_all.iloc[-5:-3, 0].values
    print(len(all_nega_phi))
    y = np.append(df_pam.loc[:'2012-06-01', 'phi_0'].values, df_ams.loc[:'2012-06-01', 'phi_0'].values)
    y = np.append(y, all_nega_phi)
    y = y[:, np.newaxis]
    # print(y)
    # print(len(x), len(y))

    # 增加交互项
    # x_int = (x[:, 0] * x[:, 1]) ** 0.5
    # x = np.column_stack((x, x_int))
    # print(x)

    model = LinearRegression(fit_intercept=True).fit(x, y)  # 构建线性模型

    R2 = model.score(x, y)  # 拟合程度 R2
    print('R2 = %.3f' % R2)  # 输出 R2
    coef = model.coef_  # 斜率
    intercept = model.intercept_  # 截距
    print(model.coef_, model.intercept_)  # 输出斜率和截距
    predicts = model.predict(x)  # 预测值
    # print(predicts.T)
    plt.scatter(range(len(y)), y)
    plt.scatter(range(len(y)), predicts)
    plt.show()
    return np.append(model.coef_, model.intercept_)


def LR_postive():
    """
    对于A大于0的情况
    """
    # time_ssn = pd.read_csv(r'./output/time_ssn.csv', header=0, index_col=0)
    # pam_ssn = time_ssn.loc['2006-08':'2014-01', 'log_ssn']
    # ams_ssn = time_ssn.loc['2011-06':'2017-03', 'log_ssn']
    # print('oooo')
    print(df_all.loc[:, 'log_ssn'])
    # df_all.loc[:, 'log_ssn'] = np.log(df_all[['ssn']].values + fix_ssn)
    all_pos_para = df_all.iloc[:-5, [3, 5]].values
    # print(all_pos_para)
    # df_pam['log_ssn'] = pam_ssn.values
    # df_ams['log_ssn'] = ams_ssn.values
    x_pam = df_pam.loc['2012-06-01':, ['log_ssn_fix', alpha_type]].values
    x_ams = df_ams.loc['2012-06-01':, ['log_ssn_fix', alpha_type]].values
    x = np.vstack((x_pam, x_ams, all_pos_para))
    # print(x)
    # scaler = StandardScaler().fit(x)
    # # print(scaler.transform(x))
    # x = scaler.transform(x)

    # x_int = (x[:, 0] * x[:, 1]) ** 0.5
    # x = np.column_stack((x, x_int))
    # print(x)

    # po = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    # x_poly = po.fit_transform(x)
    # df_nolin = pd.DataFrame(x_poly)
    # print(df_nolin)

    all_pos_phi = df_all.iloc[:-5, 0].values
    y = np.append(df_pam.loc['2012-06-01':, 'phi_0'].values, df_ams.loc['2012-06-01':, 'phi_0'].values)
    y = np.append(y, all_pos_phi)
    y = y[:, np.newaxis]

    # print(x)
    model = LinearRegression(fit_intercept=True).fit(x, y)  # 构建线性模型

    R2 = model.score(x, y)  # 拟合程度 R2
    print('R2 = %.3f' % R2)  # 输出 R2
    coef = model.coef_  # 斜率
    intercept = model.intercept_  # 截距
    print(model.coef_, model.intercept_)  # 输出斜率和截距
    predicts = model.predict(x)  # 预测值
    # print(predicts.T)

    plt.scatter(range(len(y)), y)
    plt.scatter(range(len(y)), predicts)
    plt.show()
    return np.append(model.coef_, model.intercept_)


#
def LR_all():
    x = np.vstack((df_pam.loc[:, ['log_ssn_fix', alpha_type, 'A']].values,
                   df_ams.loc[:, ['log_ssn_fix', alpha_type, 'A']].values))
    all_para = df_all.iloc[:-3, [3, 5, 7]].values
    x = np.vstack((x, all_para))

    all_phi = df_all.iloc[:-3, 0].values
    y = np.append(df_pam.loc[:, 'phi_0'].values, df_ams.loc[:, 'phi_0'].values)
    y = np.append(y, all_phi)
    y = y[:, np.newaxis]

    model = LinearRegression(fit_intercept=True).fit(x, y)  # 构建线性模型
    R2 = model.score(x, y)  # 拟合程度 R2
    print('R2 = %.3f' % R2)  # 输出 R2
    print(model.coef_, model.intercept_)
    plt.scatter(range(len(y)), y)
    plt.scatter(range(len(y)), model.predict(x))
    plt.show()
    return np.append(model.coef_, model.intercept_)


def LR_cur_all():
    def cur_fun(x_in, a, b, c, d):
        return a + b * x_in[0] * (1 + c * x_in[1]) * (1 + d * x_in[2])

    x = np.vstack((df_pam.loc[:, ['log_ssn_fix', alpha_type, 'A']].values,
                   df_ams.loc[:, ['log_ssn_fix', alpha_type, 'A']].values))
    all_para = df_all.iloc[:-3, [3, 5, 7]].values
    x = np.vstack((x, all_para)).T
    print(x[1])
    print('[][][')

    all_phi = df_all.iloc[:-3, 0].values
    y = np.append(df_pam.loc[:, 'phi_0'].values, df_ams.loc[:, 'phi_0'].values)
    y = np.append(y, all_phi)
    # y = y[:, np.newaxis]
    pp, _ = curve_fit(cur_fun, x, y,maxfev=10000)
    print(pp)

    plt.scatter(range(len(y)), y)
    plt.scatter(range(len(y)), cur_fun(x, *pp))
    plt.show()


if __name__ == "__main__":
    # para_all = LR_all()
    para_n = LR_negative()
    para_p = LR_postive()
    print('negative', para_n)
    print('postive', para_p)
    # LR_all()
    pd.DataFrame(np.vstack((para_n,para_p))).to_csv(r'./output/LR_para.csv')
    # LR_cur_all()
