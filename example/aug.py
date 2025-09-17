import pandas as pd
import sys

import h5py
import numpy as np

modal_list = ['omat24', 'mp_r2scan', 'matpes_r2scan', 'pet_mad']

def get_csv(modal):
    csv_file = open(f'{modal}.csv', 'w', buffeering = 1)
    csv_file.write('idx,formula,sgn,xx,zz\n')
    head = f'/data2_1/jinvk/25_LTC/omni/{modal}/cond'
    for i in range(25):
        if i == 7:
            continue
        data = h5py.File(f'{head}/kappa_{idx}.hdf5', 'r')[str(idx)]
        kappa = data['kappa_TOT_RTA']
        formula = kappa.attrs['formula']
        sgn = kappa.attrs['sgn']
        xx, zz = kappa[0][0][0], kappa[0][0][2]
        csv_file.write(f'{idx},{formula},{sgn},{xx},{zz}\n')
    csv_file.close()

if __name__ == '__main__':
    for modal in modal_list:
        get_csv(modal)


