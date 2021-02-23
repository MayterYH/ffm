import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time

df_ffm = pd.read_csv(r'../ssn_flux_2/output/info_test.csv', header=0, index_col=0)
df_bess = pd.read_csv(r'../input_data_0/bess_270.csv', header=0, index_col=0)
df_pam = pd.read_csv(r'../ssn_flux_2/output/pam_275.csv', header=0, index_col=0)
df_imp = pd.read_csv(r'../input_data_0/imp8_274_month.txt', header=None, index_col=None, delim_whitespace=True)

df_lr = pd.read_csv(r'../plot_Phi_4/output/info_test_lr.csv', header=0, index_col=0)

time_ffm = df_ffm.loc['1977-06-01':, 'jd'].values
flux_ffm = df_ffm.loc['1977-06-01':, 'flux_c'].values

time_pam = df_pam.iloc[:, 0].values
flux_pam = df_pam.iloc[:, 1].values

time_imp = df_imp.iloc[40:, 1].values
flux_imp = df_imp.iloc[40:, 2].values

time_bess = df_bess.iloc[:, 3].values
flux_bess = df_bess.iloc[:, 2].values

time_lr = df_lr.iloc[:, 0].values
flux_lr = df_lr.iloc[:, -2].values

df_all = pd.read_csv(r'../input_data_0/all_274.csv', header=0, index_col=0)
time_all = Time(list(df_all.index), format='iso').jd1 + 0.5
flux_all = df_all.loc[:, 'flux'].values
label_all = list(df_all.loc[:, 'app'])
# ======================================================================
plt.figure(figsize=(6, 3), dpi=300)
plt.style.use('seaborn-ticks')
plt.style.use('fast')

plt.plot(time_ffm[:], flux_ffm[:], c='gray', lw=2, label=r'$FFM$', zorder=1)
plt.plot(time_lr[:], flux_lr[:], c='lime', lw=2, label=r'$FFM_{lr}$', zorder=15, alpha=0.8)

plt.scatter(time_bess, flux_bess, s=8, marker='s', facecolors='none', edgecolor='blueviolet',
            label='BESS', zorder=3, alpha=0.8, linewidths=0.8)

plt.scatter(time_imp, flux_imp, s=8, marker='o', facecolors='none', edgecolor='deepskyblue',
            label='IMP8', zorder=2, alpha=0.8, linewidths=0.8)

plt.scatter(time_pam, flux_pam, s=8, marker='d', facecolors='none', edgecolor='darksalmon',
            label='PAMELA', zorder=2, alpha=0.8, linewidths=0.8)

for i in range(len(time_all)):
    plt.scatter(time_all[i], flux_all[i], s=8, label=label_all[i], zorder=3, alpha=0.8, linewidths=0.8)

plt.axvline(x=df_ffm.loc['1979-12-01', 'jd'], c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)
plt.axvline(x=df_ffm.loc['1989-12-01', 'jd'], c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)
plt.axvline(x=df_ffm.loc['2000-05-01', 'jd'], c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)
plt.axvline(x=df_ffm.loc['2012-06-01', 'jd'], c="k", ls="--", lw=0.8, alpha=0.3, zorder=0)

ax = plt.gca()
ax.set_title('Time - Flux (0.274 GeV)', fontsize=10, weight='bold', style='italic')
ax.set_xlabel('Time (Year)', fontsize='10')
ax.set_ylabel('Flux $(m^{2}.s.sr.GeV)^{-1}$ ', fontsize='10')
ax.tick_params(axis='both', direction='in', which='both')

plt.text(2442904.5, 3500, 'A>0', fontsize=8)
plt.text(2445636.5, 3500, 'A<0', fontsize=8)
plt.text(2449354.5, 3500, 'A>0', fontsize=8)
plt.text(2453732.5, 3500, 'A<0', fontsize=8)
# ax.set_ylim(10, 500000)
# ax.set_yscale('log')
ax.set_ylim(0, 4500)

time_label = list(Time(list(time_ffm), format='jd').iso)
x_label = [time_label[i][0:4] for i in range(len(time_label))]
plt.xticks([list(time_ffm)[i * 48 + 7] for i in range(11)], labels=[x_label[i * 48 + 7] for i in range(11)])

plt.legend(loc=1, prop={'size': 4.5})
plt.tight_layout()
plt.savefig(r'./output/flux_274_test.png', dpi=300)
plt.show()
