import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from interval import Interval
from scipy.signal import savgol_filter

import time_sequence as ts
import FUNC

"""
对 alpha 进行 月平均处理
"""

fit_ssn = 0
fit_log_ssn = 0

df_alpha = pd.read_csv(r'../input_data_0/alpha.txt', header=0, index_col=None, delim_whitespace=True)
print(df_alpha)
time = list(df_alpha.loc[:, 'Start'])
time = [time[i].replace(':', '-') for i in range(len(time))]

t_data = Time(time, format='iso').jd1
t_range = ts.generate('1976-6-1', '2020-2-1').iloc[:, 0].values
"""
截断alpha
"""
r_alpha = df_alpha.loc[:, 'R_av'].values
# r_alpha[r_alpha < 10] = 10
l_alpha = df_alpha.loc[:, 'L_av'].values
# l_alpha[l_alpha < 10] = 10

print(r_alpha)

r_mon = np.array([])
l_mon = np.array([])

for i, mon in enumerate(t_range):
    if i == len(t_range) - 1:
        break
    range_start = t_range[i]
    range_end = t_range[i + 1]
    jd_region = range_end - range_start

    zoom = Interval(range_start, range_end, upper_closed=False)

    num = 0
    count_num = np.array([])
    for j, period in enumerate(t_data):
        if (period in zoom) is True:
            print('periods : %d over' % j)
            num += 1
            count_num = np.append(count_num, j)
    if len(count_num) == 1:
        pp = int(count_num[0])
        p_0 = t_data[pp - 1]
        p_1 = t_data[pp]
        p_2 = t_data[pp + 1]

        f_1 = (p_1 - range_start) / (range_end - range_start)
        f_2 = (range_end - p_1) / (range_end - range_start)

        r_alpha_1 = r_alpha[pp - 1]
        r_alpha_2 = r_alpha[pp]

        l_alpha_1 = l_alpha[pp - 1]
        l_alpha_2 = l_alpha[pp]

        r_alpha_mon = (f_1 * r_alpha_1 + f_2 * r_alpha_2) / (f_1 + f_2)
        l_alpha_mon = (f_1 * l_alpha_1 + f_2 * l_alpha_2) / (f_1 + f_2)

        r_mon = np.append(r_mon, r_alpha_mon)
        l_mon = np.append(l_mon, l_alpha_mon)

    elif len(count_num) == 2:

        pp1 = int(count_num[0])
        pp2 = int(count_num[1])
        p_1 = t_data[pp1 - 1]
        p_2 = t_data[pp1]
        p_3 = t_data[pp2]
        p_4 = t_data[pp2 + 1]

        f_1 = (p_2 - range_start) / (range_end - range_start)
        f_2 = (p_3 - p_2) / (range_end - range_start)
        f_3 = (range_end - p_3) / (range_end - range_start)

        r_alpha_1 = r_alpha[pp1 - 1]
        r_alpha_2 = r_alpha[pp1]
        r_alpha_3 = r_alpha[pp2]

        l_alpha_1 = l_alpha[pp1 - 1]
        l_alpha_2 = l_alpha[pp1]
        l_alpha_3 = l_alpha[pp2]

        r_alpha_mon = (f_1 * r_alpha_1 + f_2 * r_alpha_2 + f_3 * r_alpha_3) / (f_1 + f_2 + f_3)
        l_alpha_mon = (f_1 * l_alpha_1 + f_2 * l_alpha_2 + f_3 * l_alpha_3) / (f_1 + f_2 + f_3)

        r_mon = np.append(r_mon, r_alpha_mon)
        l_mon = np.append(l_mon, l_alpha_mon)

    else:
        print('====================%d=====================' % i)
index = list(ts.generate('1976-6-1', '2020-1-1').index)

"""
平滑 月平均数据
"""
# print(alpha_mon)
# r_mon = np.maximum(r_mon, 7)
# l_mon = np.maximum(l_mon, 7)

r_smooth = savgol_filter(r_mon, 25, 1, mode='mirror')
l_smooth = savgol_filter(l_mon, 25, 1, mode='mirror')
r_log = np.log(r_smooth)
l_log = np.log(l_smooth)
print(r_mon)
print(r_smooth)
"""
算出A加到文件
"""
sec_1 = Interval(Time('1976-01-01', format='iso').jd1, Time('1986-01-01', format='iso').jd1,
                 lower_closed=False, upper_closed=True)
sec_2 = Interval(Time('1986-01-01', format='iso').jd1, Time('1996-01-01', format='iso').jd1,
                 lower_closed=False, upper_closed=True)
sec_3 = Interval(Time('1996-01-01', format='iso').jd1, Time('2006-01-01', format='iso').jd1,
                 lower_closed=False, upper_closed=True)
sec_4 = Interval(Time('2006-01-01', format='iso').jd1, Time('2021-01-01', format='iso').jd1,
                 lower_closed=False, upper_closed=True)
print(index)
time_A = Time([index[i] for i in range(len(index))], format='iso').jd1
A_all = np.array([])
for tt in time_A:
    if tt in sec_1:
        A_mon = FUNC.sigmoid((tt - Time('1979-12-01', format='iso').jd1) / 30)
        A_all = np.append(A_all, A_mon)
    elif tt in sec_2:
        A_mon = -FUNC.sigmoid((tt - Time('1989-12-01', format='iso').jd1) / 30)
        A_all = np.append(A_all, A_mon)
    elif tt in sec_3:
        A_mon = FUNC.sigmoid((tt - Time('2000-05-01', format='iso').jd1) / 30)
        A_all = np.append(A_all, A_mon)
    elif tt in sec_4:
        A_mon = -FUNC.sigmoid((tt - Time('2012-06-01', format='iso').jd1) / 30)
        A_all = np.append(A_all, A_mon)
A_all = np.array([format(A_all[i], '.3f') for i in range(len(A_all))], dtype='float')
print(A_all)

"""
用公式计算alpha
"""
sc20_start = Time('1965-07-01', format='iso').jd1
sc20_end = Time('1977-01-01', format='iso').jd1
gap_20 = Interval(sc20_start, sc20_end, lower_closed=True, upper_closed=False)
# print(gap_20)

sc21_start = Time('1977-01-01', format='iso').jd1
sc21_end = Time('1987-04-01', format='iso').jd1
gap_21 = Interval(sc21_start, sc21_end, lower_closed=True, upper_closed=False)

sc22_start = Time('1987-04-01', format='iso').jd1
sc22_end = Time('1996-12-01', format='iso').jd1
gap_22 = Interval(sc22_start, sc22_end, lower_closed=True, upper_closed=False)

sc23_start = Time('1996-12-01', format='iso').jd1
sc23_end = Time('2009-07-01', format='iso').jd1
gap_23 = Interval(sc23_start, sc23_end, lower_closed=True, upper_closed=False)

sc24_start = Time('2009-07-01', format='iso').jd1
sc24_end = Time('2021-01-01', format='iso').jd1
gap_24 = Interval(sc24_start, sc24_end, lower_closed=True, upper_closed=False)

time_alpha = Time([index[i] for i in range(len(index))], format='iso').jd1
# print(time_alpha)

f_alpha = np.array([])
for j, time in enumerate(time_alpha):
    if time in gap_20:
        time_in = ((time - sc20_start) / (sc20_end - sc20_start))
        result = FUNC.fun_alpha(time_in)
        f_alpha = np.append(f_alpha, result)
    elif time in gap_21:
        time_in = ((time - sc21_start) / (sc21_end - sc21_start))
        result = FUNC.fun_alpha(time_in)
        f_alpha = np.append(f_alpha, result)
    elif time in gap_22:
        time_in = ((time - sc22_start) / (sc22_end - sc22_start))
        result = FUNC.fun_alpha(time_in)
        f_alpha = np.append(f_alpha, result)
    elif time in gap_23:
        time_in = ((time - sc23_start) / (sc23_end - sc23_start))
        result = FUNC.fun_alpha(time_in)
        f_alpha = np.append(f_alpha, result)
    elif time in gap_24:
        time_in = ((time - sc24_start) / (sc24_end - sc24_start))
        result = FUNC.fun_alpha(time_in)
        f_alpha = np.append(f_alpha, result)
print(f_alpha)
"""
保存
"""
pd.DataFrame(np.vstack((r_mon, l_mon, r_smooth, l_smooth, r_log, l_log, f_alpha, A_all)).T,
             index=index, columns=['r_av', 'l_av', 'r_smooth', 'l_smooth', 'r_log', 'l_log', 'f_smooth', 'A']). \
    to_csv(r'./output/alpha_mon.csv')

df = pd.read_csv(r'./output/alpha_mon.csv', header=0, index_col=0)
r_av = df[['r_av']]
r_smooth = df[['r_smooth']]
df_ssn = pd.read_csv(r'../input_data_0/ssn_smooth_plus_test.csv', header=0, index_col=0)
ssn = df_ssn.loc['1976-06':'2020-01', 'ssn']
ssn_smooth = df_ssn.loc['1976-06':'2020-01', 'smooth']

fig = plt.figure(figsize=(7, 3), dpi=250)
plt.style.use('seaborn-ticks')
plt.style.use('fast')

ax = plt.gca()
ax1 = ax.twinx()
ax.plot(range(len(r_av)), r_av, c='dodgerblue', label=r'$\alpha$')
ax.plot(range(len(r_av)), r_smooth, c='b', label=r'$\alpha_{filt}$')
ax.plot(range(len(r_av)), f_alpha, c='m', label=r'$\alpha_{func}$')
ax.plot(range(len(r_av)), ssn, c='salmon', label=r'$SSN$')
ax.plot(range(len(r_av)), ssn_smooth, c='r', label=r'$SSN_{filt}$')

ax1.plot(range(len(r_av)), (ssn_smooth.values + fit_ssn) ** 0.5 + fit_log_ssn,
         c='lime', label=r'$\ln{SSN}$', ls='--')
ax1.plot(range(len(r_av)), A_all, c='c', label=r'$A$', ls='--')

ax.set_ylim(0, 249)
# ax1.set_ylim(0, 11)
# plt.axvline(Time('1979-12-01', format='iso').jd1, c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)
# plt.axvline(Time('1989-12-01', format='iso').jd1, c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)
# plt.axvline(Time('2000-05-01', format='iso').jd1, c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)
# plt.axvline(Time('2012-06-01', format='iso').jd1, c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)

time_label = list(df.index)
x_label = [time_label[i][:-3] for i in range(len(time_label))]
plt.xticks([range(len(time_label))[i * 48 + 7 + 12] for i in range(11)],
           labels=[x_label[i * 48 + 7 + 12] for i in range(11)])

ax.set_title('Alpha & SSN', fontsize=10, weight='bold', style='italic')
ax.set_xlabel('Time (Year)', fontsize='10', )
ax.set_ylabel('Alpha & SSN ', fontsize='10')
ax1.set_ylabel(r'$\ln{SSN}$ ', fontsize='10')
ax.tick_params(axis='both', direction='in', which='both')
ax1.tick_params(axis='both', direction='in', which='both')
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, prop={'size': 4.5})

plt.show()
