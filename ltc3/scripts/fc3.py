import numpy as np
import os, gc, sys
import sys, torch
from tqdm import tqdm
import pandas as pd

from ase import Atoms
from ase.io import read, write

from phono3py import load

from phono3py import file_IO as ph3_IO

from ltc3.util.calc import single_point_calculate_list
from ltc3.util.phonopy_utils import aseatoms2phonoatoms 

def _get_fc2_super(spg_num):
    if spg_num == 186:
        fc2_super = [5, 5, 3]

    elif spg_num in [216, 225]:
        fc2_super = [4, 4, 4]

    else:
        fc2_super = [3, 3, 3]
    return  fc2_super

def _get_fc3_super(spg_num):
    if spg_num  == 186:
        fc3_super = [3, 3, 2]

    elif spg_num in [216, 225]:
        fc3_super = [2, 2, 2]

    else:
        fc3_super = [2, 2, 2]
    return fc3_super


def _get_num_supercells(spg_num):
    if spg_num  == 186:
        num_sc = 1254

    elif spg_num == 216:
        num_sc = 222

    else:
        num_sc = 146
    return num_sc


def get_primitive_matrix(spg_num):
    if spg_num in [216, 225]:
        return [[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]

    elif spg_num == 186:
        return 'auto'

    else:
        return 'auto'

def calculate_fc3(ph3, calc, symmetrize_fc3):
    desc = 'fc3 calculation'
    forces = []
    nat = len(ph3.supercell)
    indices = []
    atoms_list = []
    for idx, sc in enumerate(ph3.supercells_with_displacements):
        if sc is not None:
            atoms_list.append(Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True))
            indices.append(idx)

    result = single_point_calculate_list(atoms_list, calc, desc=desc)

    for idx, sc in enumerate(ph3.supercells_with_displacements):
        if sc is not None:
            atoms = result[indices.index(idx)]
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    ph3.forces = force_set
    ph3.produce_fc3(symmetrize_fc3r=symmetrize_fc3)

    return ph3

def write_csv(csv_file, atoms, idx, spg_num, FC2_Error, FC3_Error):
    fc3_super = _get_fc3_super(spg_num)
    fc3_disp = _get_num_supercells(spg_num)
    formula = atoms.get_chemical_formula(empirical=True)
    csv_file.write(f'{idx},{formula},{spg_num},{fc3_super},{fc3_disp},{FC2_Error},{FC3_Error}\n')

def process_fc3(config ,calc):
    conf_fc2, conf_fc3 = config['fc2'], config['fc3']
    save_fc2, save_fc3 = conf_fc2['save'], conf_fc3['save']
    symm_fc2, symm_fc3 = conf_fc2['symm'], conf_fc3['symm']
    load_fc2, load_fc3 = conf_fc2['load'], conf_fc3['load']

    df = pd.read_csv(f'./relax_logger.csv')
    df.drop_duplicates('idx', inplace=True)
    spg_nums = list(df['sgn'])
    fc_logger = open(f'./fc3_logger.csv', 'w', buffering=1)
    fc_logger.write(f'index,formula,spg_num,prim,fc3_super,fc3_disp,fc2_error,fc3_error\n')

    for idx, spg_num in enumerate(tqdm(spg_nums, desc='processing fc3')):
        FC2_Error, FC3_Error = False, False
        atoms = read(f"{config['relax']['save']}/CONTCAR_{idx}", format='vasp')
        try:
            ph3 = load(f"{config['phonon']['save']}/phono3py_params_fc2_{idx}.yaml")

        except Exception as e:
            FC2_Error = True
            print(e)
            write_csv(fc_logger, atoms, idx, spg_num, FC2_Error, FC3_Error)
            continue

        try:
            ph3 = calculate_fc3(ph3, calc, symmetrize_fc3=symm_fc3)
            ph3_IO.write_fc3_to_hdf5(
                ph3.fc3,
                filename=f'{save_fc3}/fc3_{idx}.hdf5',
            )
        except Exception as e:
            sys.stderr.write(f'FC3 calc error at {idx}: {e}\n')
            FC3_Error = True
            write_csv(fc_logger, atoms, idx, spg_num, FC2_Error, FC3_Error)
            continue

        write_csv(fc_logger, atoms, idx, spg_num, FC2_Error, FC3_Error)
        ph3.save(f'{config["phonon"]["save"]}/phono3py_params_fc3_{idx}.yaml', settings={'compress': True})
        del atoms, ph3
        torch.cuda.empty_cache()
        gc.collect()
    fc_logger.close()
