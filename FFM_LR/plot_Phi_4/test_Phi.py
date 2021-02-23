import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import FUNC

fix_ssn = 0
fix_log_ssn = 0
fix_alpha = 0

alpha_type = 'r_smooth'

df_info = pd.read_csv(r'../ssn_flux_2/output/info_test.csv', header=0, index_col=0)
df_alpha = pd.read_csv(r'../mon_alpha_1/output/alpha_mon.csv', header=0, index_col=0)
print(df_alpha)

phi_c = df_info.loc['1977-06-01':'2020-01-01', 'phi_c'].values
Phi_c = df_info.loc['1977-06-01':'2020-01-01', 'Phi_c'].values
ssn_delay = (df_info.loc['1977-06-01':'2020-01-01', 'ssn_delay'].values + fix_ssn) ** 0.5 + fix_log_ssn
print(ssn_delay)

alpha = df_alpha.loc[:, alpha_type].values
dealy = np.ones(len(alpha) - 12, dtype='int') * -12

# dealy[:24] = np.ones(24, dtype='int') * -8

alpha_dealy = FUNC.delay_ssn(alpha, dealy) + fix_alpha

A_values = df_alpha.loc['1977-05-01':, 'A'].values
# alpha_dealy = np.log(alpha_dealy)

print(len(A_values))

mark = df_info.loc['1977-06-01':'2020-01-01', 'mark'].values
para = np.vstack((ssn_delay, alpha_dealy)).T

phi_lr = np.array([])

# for i in range(len(mark)):
#     phi = FUNC.LR_all(para[i])
#     phi_lr = np.append(phi_lr, phi)

for i, flag in enumerate(mark):
    if flag == -12:
        phi = FUNC.LR_negative(para[i])
        phi_lr = np.append(phi_lr, phi)
    elif flag == -5:
        phi = FUNC.LR_postive(para[i])
        phi_lr = np.append(phi_lr, phi)


print(phi_lr)
print(len(phi_lr))


"""
保存LR计算出的phi_lr 计算 flux_lr (log)
"""
df_info_lr = df_info.loc['1977-06-01':'2020-01-01', :]
print(len(df_info_lr))

df_info_lr['phi_lr'] = phi_lr
print(df_info_lr)
df_info_lr.plot(y='phi_lr')

b_c = df_info.loc['1977-06-01':'2020-01-01', 'b_c'].values
Phi_lr = FUNC.fit_obj_vary(0.274, phi_lr, b_c)
flux_lr = FUNC.ffm_fun_all(0.274 * np.ones(len(Phi_lr)), Phi_lr)
# ==============================================================
'''
平滑flux
'''
print('==============================================================')

phi_pos = np.array([FUNC.LR_postive(para[i]) for i in range(len(para))])
Phi_pos = FUNC.fit_obj_vary(0.274, phi_pos, b_c)
flux_pos = FUNC.ffm_fun_all(0.274 * np.ones(len(Phi_pos)), Phi_pos)

phi_neg = np.array([FUNC.LR_negative(para[i]) for i in range(len(para))])
Phi_neg = FUNC.fit_obj_vary(0.274, phi_neg, b_c)
flux_neg = FUNC.ffm_fun_all(0.274 * np.ones(len(Phi_neg)), Phi_neg)
df_flux = pd.DataFrame(np.vstack((flux_pos, flux_neg, Phi_pos, Phi_neg)).T,
                       index=list(df_info_lr.index), columns=['pos', 'neg', 'phi_pos', 'phi_neg'])

# plt.plot(range(len(Phi_pos)),Phi_lr)
# # plt.plot(range(len(Phi_pos)),Phi_neg)
# # df_flux.plot(y=['pos','neg'])
# plt.show()
# print(df_flux)
p1, p2 = FUNC.sigmod_smooth(24)
# df_flux.to_csv(r'./output/flux.csv')

fp1 = df_flux.loc['1977-12-01':'1981-12-01', 'pos']
fn1 = df_flux.loc['1977-12-01':'1981-12-01', 'neg']
f1 = (fp1 * p1 + fn1 * p2)

fp2 = df_flux.loc['1987-12-01':'1991-12-01', 'pos']
fn2 = df_flux.loc['1987-12-01':'1991-12-01', 'neg']
f2 = (fn2 * p1 + fp2 * p2)

fp3 = df_flux.loc['1998-05-01':'2002-05-01', 'pos']
fn3 = df_flux.loc['1998-05-01':'2002-05-01', 'neg']
f3 = (fp3 * p1 + fn3 * p2)

fp4 = df_flux.loc['2010-06-01':'2014-06-01', 'pos']
fn4 = df_flux.loc['2010-06-01':'2014-06-01', 'neg']
f4 = (fn4 * p1 + fp4 * p2)

"""
平滑Phi_lr
"""
# pp1 = df_flux.loc['1977-12-01':'1981-12-01', 'phi_pos']
# pn1 = df_flux.loc['1977-12-01':'1981-12-01', 'phi_neg']
# ppp1 = (pp1 * p1 + pn1 * p2)
# f1 = FUNC.ffm_fun_all(0.274 * np.ones(len(ppp1)), ppp1)
#
# pp2 = df_flux.loc['1987-12-01':'1991-12-01', 'phi_pos']
# pn2 = df_flux.loc['1987-12-01':'1991-12-01', 'phi_neg']
# ppp2 = (pn2 * p1 + pp2 * p2)
# f2 = FUNC.ffm_fun_all(0.274 * np.ones(len(ppp2)), ppp2)
#
# pp3 = df_flux.loc['1998-05-01':'2002-05-01', 'phi_pos']
# pn3 = df_flux.loc['1998-05-01':'2002-05-01', 'phi_neg']
# ppp3 = (pp3 * p1 + pn3 * p2)
# f3 = FUNC.ffm_fun_all(0.274 * np.ones(len(ppp3)), ppp3)
#
# pp4 = df_flux.loc['2010-06-01':'2014-06-01', 'phi_pos']
# pn4 = df_flux.loc['2010-06-01':'2014-06-01', 'phi_neg']
# ppp4 = (pn4 * p1 + pp4 * p2)
# f4 = FUNC.ffm_fun_all(0.274 * np.ones(len(ppp4)), ppp4)
print('==============================================================')

df_info_lr['flux_lr'] = flux_lr
df_info_lr['r_smooth'] = alpha[12:]
# df_info_lr.plot(y='flux_lr',c='r')
print('===============================================================')
# print(df_info_lr)
df_info_lr.loc['1977-12-01':'1981-12-01', 'flux_lr'] = f1
df_info_lr.loc['1987-12-01':'1991-12-01', 'flux_lr'] = f2
df_info_lr.loc['1998-05-01':'2002-05-01', 'flux_lr'] = f3
df_info_lr.loc['2010-06-01':'2014-06-01', 'flux_lr'] = f4

df_info_lr.plot(y='flux_lr')
plt.show()
# # print(f4)
print('===============================================================')
df_info_lr.to_csv(r'./output/info_test_lr.csv')

time = list(df_info.loc['1977-06-01':'2020-01-01', 'jd'].values)
label = list(df_info.loc['1977-06-01':'2020-01-01', 'jd'].index)

x_label = [label[i * 4 * 12 + 7][:-6] for i in range(11)]
x_tickl = [time[i * 4 * 12 + 7] for i in range(11)]
plt.figure(figsize=(6, 3), dpi=300)
plt.style.use('seaborn-ticks')
plt.style.use('fast')

plt.scatter(time, phi_c, s=5, marker='o', facecolors='none', edgecolor='dodgerblue',
            label=r'$\phi_{c}$', zorder=3, alpha=0.8, linewidths=0.8)

plt.scatter(time, phi_lr, s=5, marker='o', facecolors='none', edgecolor='darksalmon',
            label=r'$\phi_{lr}$', zorder=3, alpha=0.8, linewidths=0.8)

plt.plot(time, Phi_c, label=r'$\Phi_{c}$', zorder=3, alpha=0.8, lw=1.8, c='b')
plt.plot(time, Phi_lr, c='r', label=r'$\Phi_{lr}$', zorder=3, alpha=0.8, lw=1.8)

plt.axvline(x=df_info.loc['1979-12-01', 'jd'], c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)
plt.axvline(x=df_info.loc['1989-12-01', 'jd'], c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)
plt.axvline(x=df_info.loc['2000-05-01', 'jd'], c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)
plt.axvline(x=df_info.loc['2014-01-01', 'jd'], c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)

ax = plt.gca()
ax.set_title(r'Time - $\phi$', fontsize=10, weight='bold', style='italic')
ax.set_xlabel('Time (Year)', fontsize='10')
ax.set_ylabel(r'$\phi$ ', fontsize='10')
ax.tick_params(axis='both', direction='in', which='both')

plt.text(2442904.5, 1.5, 'A>0', fontsize=8)
plt.text(2445636.5, 1.5, 'A<0', fontsize=8)
plt.text(2449354.5, 1.5, 'A>0', fontsize=8)
plt.text(2453732.5, 1.5, 'A<0', fontsize=8)
# ax.set_ylim(0, 2)
plt.xticks(ticks=x_tickl, labels=x_label)
plt.legend(loc=1, prop={'size': 6})
plt.tight_layout()
plt.show()
