import os, gc, sys, h5py, pickle
from tqdm import tqdm
from phono3py import load
from phono3py import file_IO as ph3_IO
import pandas as pd
import h5py
from copy import deepcopy
import numpy as np
import time

def check_imaginary_freqs(frequencies: np.ndarray) -> bool:
    try:
        if np.all(np.isnan(frequencies)):
            return True

        if np.any(frequencies[0, 3:] < 0):
            return True

        if np.any(frequencies[0, :3] < -1e-2):
            return True

        if np.any(frequencies[1:] < 0):
            return True
    except Exception as e:
        print(f"Failed to check imaginary frequencies: {e!r}")

    return False

def _get_mesh(spg_num):
    if spg_num == 186:
        mesh = [19, 19, 15]
    else:
        mesh = [19, 19, 19]
    return mesh

def postprocess_kappa_to_csv(file, idx, temps, kappas):
    for temp, kappa in zip(temps, kappas):
        if kappa is None:
            kappa_join = 'NaN'
        else:
            kappa = kappa.reshape(-1)
            kappa_join = ','.join(map(str,kappa))
        file.write(f'{idx},{temp},{kappa_join}\n')

def process_conductivity(head):
    save_dir = f'{head}/five'
    conductivity_type = 'wigner'

    df = pd.read_csv(f'{head}/relax_logger.csv')
    df.drop_duplicates('idx', inplace=True)
    spg_nums = list(df['sgn'])
 
    csv_tot = open(f'{save_dir}/kappa_total.csv', 'w', buffering=1)
    csv_tot.write(f'index,temperature,xx,yy,zz,yz,xz,xy\n')

    csv_p = open(os.path.join(save_dir,'kappa_p.csv'), 'w', buffering=1)
    csv_p.write(f'index,temperature,xx,yy,zz,yz,xz,xy\n')
    csv_c = open(os.path.join(save_dir,'kappa_c.csv'), 'w', buffering=1)
    csv_c.write(f'index,temperature,xx,yy,zz,yz,xz,xy\n')


    cond_keys = ['frequency', 'gamma', 'gamma_isotope', 'grid_point_count', 'grid_points', 'grid_weights', 'gv_by_gv_operator', 'kappa_C', 'kappa_P_RTA', 'kappa_TOT_RTA', 'mode_heat_capacities', 'mode_kappa_C', 'mode_kappa_P_RTA', 'number_of_ignored_phonon_modes', 'qpoints', 'temperatures', 'velocity_operator']

    KAPPA_KEYS = ['kappa_TOT_RTA', 'kappa_P_RTA', 'kappa_C']
    load_fc2, load_fc3, load_ph3 = f'{head}/fc2', f'{head}/fc3', f'{head}/phonon'
    temperatures = [300,]
    idx_list = [59, 97]
    spg_num_list = [186, 216]

    for idx, spg_num in tqdm(zip(idx_list, spg_num_list), desc='calculating conductivity'):
        mesh = _get_mesh(spg_num)
        ph3 = load(f'{load_ph3}/phono3py_params_fc2_{idx}.yaml')
        fc3 = ph3_IO.read_fc3_from_hdf5(f'{load_fc3}/fc3_{idx}.hdf5')
        ph3.fc3 = fc3

        ph3.mesh_numbers = mesh

        ph3.init_phph_interaction(symmetrize_fc3q=False)

        ph3.run_thermal_conductivity(
            temperatures=temperatures,
            conductivity_type='wigner',
            is_isotope=False
            )

        kappa = ph3.thermal_conductivity
        ph3.save(f'{save_dir}/phono3py_{idx}.yaml')

        with open(f'{save_dir}/kappa_{idx}.pkl', 'wb') as f:
            pickle.dump(kappa, f)

        with h5py.File(f'{save_dir}/kappa_{idx}.hdf5', 'w') as f:
            g = f.create_group(f'{idx}')
            for key in cond_keys:
                try:
                    g.create_dataset(key, data = getattr(kappa, key))
                except:
                    continue
            g.attrs['spg_num'] = spg_num
            g.attrs['formula'] = ph3.unitcell.formula
            g.attrs['idx'] = idx
            g.attrs['type'] = 'wigner'

        try:
           cond_dict = {
               'kappa_TOT_RTA': deepcopy(getattr(kappa, 'kappa_TOT_RTA')),
               'kappa_P_RTA': deepcopy(getattr(kappa, 'kappa_P_RTA')),
               'kappa_C': deepcopy(getattr(kappa, 'kappa_C')),
            }
            
        except Exception as e:
            sys.stderr.write(f'Conductivity error in {idx}: {e}\n')
            nones = [None for _ in temperatures]
            cond_dict = {key: nones for key in KAPPA_KEYS}

        try:
            total_key = 'kappa_TOT_RTA' 
            postprocess_kappa_to_csv(csv_tot, idx, temperatures, cond_dict[total_key])
            postprocess_kappa_to_csv(
                csv_p, idx, temperatures, cond_dict['kappa_P_RTA'],
            )
            postprocess_kappa_to_csv(csv_c, idx, temperatures, cond_dict['kappa_C'])

        except Exception as exc:
            print(exc)

        del ph3
        gc.collect()
        time.sleep(4)

    csv_tot.close()
    csv_p.close()
    csv_c.close()


if __name__ == '__main__':
    process_conductivity(head = sys.argv[1])
