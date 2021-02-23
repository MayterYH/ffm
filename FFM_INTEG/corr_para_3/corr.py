import pandas as pd
import numpy as np
import scipy.stats

df_pam = pd.read_csv(r'../double_two_2/output/fill_all/pamela_all.csv', header=0, index_col=0)
df_ams = pd.read_csv(r'../double_two_2/output/fill_all/ams02_all.csv', header=0, index_col=0)
df_ssn = pd.read_csv(r'../input_data_0/ssn_smooth/ssn_smooth_plus.csv', header=0, index_col=0)


def pam_corr():
    phi = df_pam.iloc[:, 0].values
    b = df_pam.iloc[:, 1].values
    ssn = df_ssn.loc[:'2014-01', 'smooth'].values

    phi_ssn = np.array([])
    b_ssn = np.array([])
    for j in range(20):
        if j == 0:
            phi_r, pp = scipy.stats.pearsonr(phi, ssn[-90:])
            b_r, pb = scipy.stats.pearsonr(b, ssn[-90:])
        else:
            phi_r, pp = scipy.stats.pearsonr(phi, ssn[-90 - j:-j])
            b_r, pb = scipy.stats.pearsonr(b, ssn[-90 - j:-j])
        phi_ssn = np.append(phi_ssn, phi_r)
        b_ssn = np.append(b_ssn, b_r)
    x_max = np.where(abs(phi_ssn) == np.max(abs(phi_ssn)))
    y_max = abs(phi_ssn).max()
    print(x_max, y_max)
    pd.DataFrame(np.vstack((phi_ssn, b_ssn)).T, index=range(20),
                 columns=['phi_ssn', 'b_ssn']).to_csv(r'./output/pam-coor-x%d.csv' % int(x_max[0]))


def ams_corr():
    phi = df_ams.iloc[:, 0].values
    b = df_ams.iloc[:, 1].values
    ssn = df_ssn.loc[:'2017-03', 'smooth'].values

    phi_ssn = np.array([])
    b_ssn = np.array([])
    for j in range(20):
        if j == 0:
            phi_r, pp = scipy.stats.pearsonr(phi, ssn[-70:])
            b_r, pb = scipy.stats.pearsonr(b, ssn[-70:])
        else:
            phi_r, pp = scipy.stats.pearsonr(phi, ssn[-70 - j:-j])
            b_r, pb = scipy.stats.pearsonr(b, ssn[-70 - j:-j])
        phi_ssn = np.append(phi_ssn, phi_r)
        b_ssn = np.append(b_ssn, b_r)
    x_max = np.where(abs(phi_ssn) == np.max(abs(phi_ssn)))
    y_max = abs(phi_ssn).max()
    print(x_max, y_max)
    pd.DataFrame(np.vstack((phi_ssn, b_ssn)).T, index=range(20),
                 columns=['phi_ssn', 'b_ssn']).to_csv(r'./output/ams-coor-x%d.csv' % int(x_max[0]))


if __name__ == "__main__":
    pam_corr()
    ams_corr()
