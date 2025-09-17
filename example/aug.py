import pandas as pd
import sys

import h5py
import numpy as np

modal_list = ['omat24', 'mp_r2scan', 'matpes_r2scan', 'pet_mad']

def get_csv(modal):
    csv_file = open(f'{modal}.csv', 'w', buffering = 1)
    csv_file.write('idx,formula,sgn,xx,zz\n')
    head = f'/data2_1/jinvk/25_LTC/omni/{modal}/cond'
    for i in range(25):
        if i == 7:
            continue
        file = h5py.File(f'{head}/kappa_{i}.hdf5', 'r')
        data = file[str(i)]
        kappa = data['kappa_TOT_RTA']
        formula = data.attrs['formula']
        sgn = data.attrs['sgn']
        xx, zz = kappa[0][0][0], kappa[0][0][2]
        file.close()
        csv_file.write(f'{i},{formula},{sgn},{xx},{zz}\n')
    csv_file.close()

if __name__ == '__main__':
    for modal in modal_list:
        get_csv(modal)


