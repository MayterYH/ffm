import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import FUNC


def plot_pam():
    pamSigphi = pd.read_csv(r'../ffm_Phi_1/output/pam_phi_sig.csv', header=0, index_col=0)
    pamTwopara = pd.read_csv(r'../sep_find_line_4/output/pam_info.csv', header=0, index_col=0)
    file = os.listdir(r'../input_data_0/pam_mon/')
    for i, name in enumerate(file):
        # 观测数据
        df = pd.read_csv(r'../input_data_0/pam_mon/%s' % name, header=0, index_col=0)
        e = df.index.values
        flux = df.iloc[:, 0]
        # 不随能量变化的 phi 计算 出的通量
        e_series = np.linspace(0.0085, 50, 1000)
        fluxSigpoint = FUNC.ffm_fun_all(e, pamSigphi.iloc[:, 0].values[i])
        fluxSigseries = FUNC.ffm_fun_all(e_series, pamSigphi.iloc[:, 0].values[i])

        # 模型计算出的  通量
        Phi_point_c = FUNC.fit_obj_vary(e, pamTwopara.loc[:, 'phi_0'].values[i],
                                        pamTwopara.loc[:, 'b'].values[i])
        fluxModlepoint = FUNC.ffm_fun_all(e, Phi_point_c)

        Phi_serier_c = FUNC.fit_obj_vary(e_series, pamTwopara.loc[:, 'phi_0'].values[i],
                                         pamTwopara.loc[:, 'b'].values[i])
        fluxModleseries = FUNC.ffm_fun_all(e_series, Phi_serier_c)

        # SSN 计算出的结果
        Phi_point_ssn = FUNC.fit_obj_vary(e, pamTwopara.loc[:, 'phi_c'].values[i],
                                          pamTwopara.loc[:, 'b_c'].values[i])
        fluxSSNpoint = FUNC.ffm_fun_all(e, Phi_point_ssn)

        Phi_series_ssn = FUNC.fit_obj_vary(e_series, pamTwopara.loc[:, 'phi_c'].values[i],
                                           pamTwopara.loc[:, 'b_c'].values[i])
        fluxSSNseries = FUNC.ffm_fun_all(e_series, Phi_series_ssn)
        # LIS
        flux_lis = FUNC.LIS_5(e_series)
        # 误差
        sig_err_point = (fluxSigpoint - flux) / flux
        mod_err_point = (fluxModlepoint - flux) / flux
        ssn_err_point = (fluxSSNpoint - flux) / flux

        # 画图
        plt.figure(figsize=(8, 6))
        plt.style.use('fast')
        plt.style.use('seaborn-ticks')
        grid = plt.GridSpec(8, 6, hspace=0)
        ax_flux = plt.subplot(grid[0:5, 0:6])

        plt.scatter(e, flux, marker='o', label='PAMELA', facecolors='none', edgecolor='k', alpha=0.8, s=40,
                    zorder=100)
        plt.scatter(e, fluxSSNpoint, marker='D', s=15, facecolors='none', edgecolor='orange', alpha=1, label='FFM',
                    zorder=101)
        plt.plot(e_series, flux_lis, 'r--', color='m', label='LIS', lw=2.5, zorder=10, alpha=0.5)
        plt.plot(e_series, fluxSigseries, 'r-', color='dodgerblue', label='FIT', alpha=0.8, lw=2.5, zorder=11)
        plt.plot(e_series, fluxModleseries, color='lime', label='MODEL', alpha=0.8, lw=2.5, zorder=1)

        ax_err = plt.subplot(grid[5:8, 0:6], sharex=ax_flux)
        plt.scatter(e, ssn_err_point * 100, marker='D', facecolor='none', edgecolors='orange', label='FFM', s=15,
                    alpha=0.8)
        plt.scatter(e, sig_err_point * 100, marker='o', facecolor='none', edgecolors='dodgerblue', label='FIT', s=15,
                    alpha=0.8)
        plt.scatter(e, mod_err_point * 100, marker='o', facecolor='none', edgecolors='lime', label='MODEL', s=15,
                    alpha=0.8)

        ax_err.set_ylim(-30, 30)
        ax_err.grid(True, ls='-.', which='both', alpha=0.2)
        ax_err.tick_params(axis='both', direction='in', which='both')
        ax_err.set_ylabel(r'$(F_{pre}-F_{obs})/F_{obs}$ (%)', fontsize='10')
        ax_err.legend(loc='upper right')

        # ######################################################################
        periods_list = list(pamTwopara.index)
        ax_flux.set_title('E-Flux Periods: %s' % periods_list[i],
                          fontsize=10, weight='bold', style='italic')
        ax_err.set_xlabel('Kinetic Energy (GeV)', fontsize='10')
        ax_flux.set_ylabel('Flux $(m^{2}.s.sr.GeV)^{-1}$ ', fontsize='10')
        ax_flux.set_xscale('log')

        ax_flux.set_yscale('log')
        ax_err.set_xlim(0.0885, 115)
        ax_flux.set_ylim(0.2, 30000)
        ax_flux.tick_params(axis='both', direction='in', which='both')
        ax_flux.legend(loc='upper right')
        plt.savefig(r'./output/flux/pamela/periods_%s.png' % periods_list[i], dpi=250)
        plt.savefig(r'./output/flux/pamela/gif/periods_%d.png' % i, dpi=250)  # 保存路径;
        plt.cla()
        plt.close("all")
        print('PAMELA %s' % periods_list[i], '完成 ==>')
    print('PAMELA 结束')


def plot_ams():
    amsSigphi = pd.read_csv(r'../ffm_Phi_1/output/ams_phi_sig.csv', header=0, index_col=0)
    amsTwopara = pd.read_csv(r'../sep_find_line_4/output/ams_info.csv', header=0, index_col=0)
    file = os.listdir(r'../input_data_0/ams_mon/')
    for i, name in enumerate(file):
        # 观测数据
        df = pd.read_csv(r'../input_data_0/ams_mon/%s' % name, header=0, index_col=0)
        e = df.index.values
        flux = df.iloc[:, 0]
        # 不随能量变化的 phi 计算 出的通量
        e_series = np.linspace(0.48, 60, 1000)
        fluxSigpoint = FUNC.ffm_fun_all(e, amsSigphi.iloc[:, 0].values[i])
        fluxSigseries = FUNC.ffm_fun_all(e_series, amsSigphi.iloc[:, 0].values[i])

        # 模型计算出的  通量
        Phi_point_c = FUNC.fit_obj_vary(e, amsTwopara.loc[:, 'phi_0'].values[i],
                                        amsTwopara.loc[:, 'b'].values[i])
        fluxModlepoint = FUNC.ffm_fun_all(e, Phi_point_c)

        Phi_serier_c = FUNC.fit_obj_vary(e_series, amsTwopara.loc[:, 'phi_0'].values[i],
                                         amsTwopara.loc[:, 'b'].values[i])
        fluxModleseries = FUNC.ffm_fun_all(e_series, Phi_serier_c)

        # SSN 计算出的结果
        Phi_point_ssn = FUNC.fit_obj_vary(e, amsTwopara.loc[:, 'phi_c'].values[i],
                                          amsTwopara.loc[:, 'b_c'].values[i])
        fluxSSNpoint = FUNC.ffm_fun_all(e, Phi_point_ssn)

        Phi_series_ssn = FUNC.fit_obj_vary(e_series, amsTwopara.loc[:, 'phi_c'].values[i],
                                           amsTwopara.loc[:, 'b_c'].values[i])
        fluxSSNseries = FUNC.ffm_fun_all(e_series, Phi_series_ssn)
        # LIS
        flux_lis = FUNC.LIS_5(e_series)
        # 误差
        sig_err_point = (fluxSigpoint - flux) / flux
        mod_err_point = (fluxModlepoint - flux) / flux
        ssn_err_point = (fluxSSNpoint - flux) / flux

        # 画图
        plt.figure(figsize=(8, 6))
        plt.style.use('fast')
        plt.style.use('seaborn-ticks')
        grid = plt.GridSpec(8, 6, hspace=0)
        ax_flux = plt.subplot(grid[0:5, 0:6])

        plt.scatter(e, flux, marker='o', label='AMS02', facecolors='none', edgecolor='k', alpha=0.8, s=40,
                    zorder=100)
        plt.scatter(e, fluxSSNpoint, marker='D', s=15, facecolors='none', edgecolor='orange', alpha=1, label='FFM',
                    zorder=101)
        plt.plot(e_series, flux_lis, 'r--', color='m', label='LIS', lw=2.5, zorder=10, alpha=0.5)
        plt.plot(e_series, fluxSigseries, 'r-', color='dodgerblue', label='FIT', alpha=0.8, lw=2.5, zorder=11)
        plt.plot(e_series, fluxModleseries, color='lime', label='MODEL', alpha=0.8, lw=2.5, zorder=1)

        ax_err = plt.subplot(grid[5:8, 0:6], sharex=ax_flux)
        plt.scatter(e, ssn_err_point * 100, marker='D', facecolor='none', edgecolors='orange', label='FFM', s=15,
                    alpha=0.8)
        plt.scatter(e, sig_err_point * 100, marker='o', facecolor='none', edgecolors='dodgerblue', label='FIT', s=15,
                    alpha=0.8)
        plt.scatter(e, mod_err_point * 100, marker='o', facecolor='none', edgecolors='lime', label='MODEL', s=15,
                    alpha=0.8)

        ax_err.set_ylim(-30, 30)
        ax_err.grid(True, ls='-.', which='both', alpha=0.2)
        ax_err.tick_params(axis='both', direction='in', which='both')
        ax_err.set_ylabel(r'$(F_{pre}-F_{obs})/F_{obs}$ (%)', fontsize='10')
        ax_err.legend(loc='upper right')

        # ######################################################################
        periods_list = list(amsTwopara.index)
        ax_flux.set_title('E-Flux Periods: %s' % periods_list[i],
                          fontsize=10, weight='bold', style='italic')
        ax_err.set_xlabel('Kinetic Energy (GeV)', fontsize='10')
        ax_flux.set_ylabel('Flux $(m^{2}.s.sr.GeV)^{-1}$ ', fontsize='10')
        ax_flux.set_xscale('log')

        ax_flux.set_yscale('log')
        ax_err.set_xlim(0.48, 115)
        ax_flux.set_ylim(0.15, 30000)
        ax_flux.tick_params(axis='both', direction='in', which='both')
        ax_flux.legend(loc='upper right')
        plt.savefig(r'./output/flux/ams02/periods_%s.png' % periods_list[i], dpi=250)  # 保存路径;
        plt.cla()
        plt.close("all")
        print('AMS02 %s' % periods_list[i], '完成 ==>')
    print('AMS02 结束')


if __name__ == "__main__":
    plot_pam()
    plot_ams()
