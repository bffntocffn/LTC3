import os, gc
import sys, torch
from tqdm import tqdm
from ase.io import read
from phono3py import Phono3py, load
from phono3py import file_IO as ph3_IO
import pandas as pd
from ltc3.util.phonopy_utils import check_imaginary_freqs

def _get_mesh(spg_num):
    if spg_num == 186:
        mesh = [19, 19, 15]
    else:
        mesh = [19, 19, 19]
    return mesh

def postprocess_kappa_to_csv(file, idx, temps, kappas, mesh, Im):
    for temp, kappa in zip(temps, kappas):
        if kappa is None:
            kappa_join = 'NaN'
        else:
            kappa = kappa.reshape(-1)
            kappa_join = ','.join(map(str,kappa))
        file.write(f'{idx},{temp},{kappa_join},{mesh},{Im}\n')

def process_conductivity(config):
    conf = config['cond']
    save_dir = conf['save']

    if isinstance(temp := conf['temperature'], list):
        temperatures = list(range(temp[0], temp[1]+1, temp[2]))
    else:
        temperatures = [temp]

    if conf['cond_type'].lower() == 'bte':
        conductivity_type = None
    else:
        conductivity_type = 'wigner'

    df = pd.read_csv(f'./relax_logger.csv')
    df.drop_duplicates('idx', inplace=True)
    spg_nums = list(df['sgn'])
 
    csv_tot = open(f'{save_dir}/kappa_total.csv', 'w', buffering=1)
    csv_tot.write(f'index,temperature,xx,yy,zz,yz,xz,xy,mesh,Imaginary\n')
    csv_p, csv_c = None, None

    if conductivity_type == 'wigner':
        # bte method doesn't need this? idk sadly this is way behind my priorities
        csv_p = open(os.path.join(save_dir,'kappa_p.csv'), 'w', buffering=1)
        csv_p.write(f'index,temperature,xx,yy,zz,yz,xz,xy\n')
        csv_c = open(os.path.join(save_dir,'kappa_c.csv'), 'w', buffering=1)
        csv_c.write(f'index,temperature,xx,yy,zz,yz,xz,xy\n')


    KAPPA_KEYS = ['kappa', 'kappa_TOT_RTA', 'kappa_P_RTA', 'kappa_C']
    load_fc2, load_fc3 = config['fc2']['save'], config['fc3']['save']

    for idx, spg_num in tqdm(enumerate(spg_nums), desc='calculating conductivity'):
        Im = False
        mesh = _get_mesh(spg_num)
        ph3 = load(f'{config["phonon"]["save"]}/phono3py_params_fc2_{idx}.yaml')
        fc3 = ph3_IO.read_fc3_from_hdf5(f'{load_fc3}/fc3_{idx}.hdf5')
        ph3.fc3 = fc3

        ph3.mesh_numbers = mesh
        print(f'index .. {idx}')
        print(f'mesh numbers .. {mesh}')

        try:
            ph3.init_phph_interaction(symmetrize_fc3q=False)
            ph3.run_phonon_solver()
            freqs, eigvecs, grid = ph3.get_phonon_data()
            has_imag = check_imaginary_freqs(freqs)
            if has_imag:
                Im = True
                print(f'{idx}-th structure {atoms} has imaginary frequencies!')

            ph3.run_thermal_conductivity(
                temperatures=temperatures,
                conductivity_type=conductivity_type
            )
            cond = ph3.thermal_conductivity
            cond_dict = {
                k: getattr(cond, k) for k in KAPPA_KEYS if hasattr(cond, k)
            }
            
        except Exception as e:
            sys.stderr.write(f'Conductivity error in {idx}: {e}\n')
            nones = [None for _ in temperatures]
            cond_dict = {key: nones for key in KAPPA_KEYS}

        total_key = 'kappa_TOT_RTA' if conductivity_type == 'wigner' else 'kappa'
        postprocess_kappa_to_csv(csv_tot, idx, temperatures, cond_dict[total_key], mesh, Im)
        if conductivity_type == 'wigner':
            postprocess_kappa_to_csv(
                csv_p, idx, temperatures, cond_dict['kappa_P_RTA'], mesh, Im,
            )
            postprocess_kappa_to_csv(csv_c, idx, temperatures, cond_dict['kappa_C'], mesh, Im)
        del ph3
        gc.collect()
    csv_tot.close()
    if conductivity_type == 'wigner':
        csv_p.close()
        csv_c.close()
