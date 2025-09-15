import numpy as np
import os, gc, sys
import sys, torch
from tqdm import tqdm
import pandas as pd

from ase import Atoms
from ase.io import read, write

from phono3py import Phono3py
from phonopy import file_IO as ph_IO

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


def get_primitive_matrix(spg_num):
    if spg_num in [216, 225]:
        return [[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]

    elif spg_num == 186:
        return 'auto'

    else:
        return 'auto'

def calculate_fc2(ph3, calc, symmetrize_fc2):
    desc = 'fc2 calculation'
    forces = []
    nat = len(ph3.phonon_supercell)
    indices = []
    atoms_list = []
    for idx, sc in enumerate(ph3.phonon_supercells_with_displacements):
        if sc is not None:
            atoms_list.append(Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True))
            indices.append(idx)

    result = single_point_calculate_list(atoms_list, calc, desc=desc)

    for idx, sc in enumerate(ph3.phonon_supercells_with_displacements):
        if sc is not None:
            atoms = result[indices.index(idx)]
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    ph3.phonon_forces = force_set
    ph3.produce_fc2(symmetrize_fc2=symmetrize_fc2)

    return ph3

def write_csv(csv_file, atoms, ph3, idx, spg_num, FC2_Error):
    fc2_super = _get_fc2_super(spg_num)
    prim_matrix = get_primitive_matrix(spg_num)
    fc2_disp = len(ph3.phonon_supercells_with_displacements)
    formula = atoms.get_chemical_formula(empirical=True)
    csv_file.write(f'{idx},{formula},{spg_num},{prim_matrix},{fc2_super},{fc2_disp},{FC2_Error}\n')

def process_fc2(config ,calc):
    conf_fc2, conf_fc3 = config['fc2'], config['fc3']
    save_fc2 = conf_fc2['save']
    symm_fc2 = conf_fc2['symm']
    load_fc2 = conf_fc2['load']

    df = pd.read_csv(f'./relax_logger.csv')
    df.drop_duplicates('idx', inplace=True)
    spg_nums = list(df['sgn'])
    fc_logger = open(f'./fc2_logger.csv', 'w', buffering=1)
    fc_logger.write(f'index,formula,spg_num,prim,fc2_super,fc2_disp,fc2_error\n')

    for idx, spg_num in enumerate(tqdm(spg_nums, desc='processing fc2s')):
        FC2_Error  = False
        atoms = read(f"{config['relax']['save']}/CONTCAR_{idx}", format='vasp')
        unit_cell = aseatoms2phonoatoms(atoms)
        fc2_super = _get_fc2_super(spg_num)
        fc3_super = _get_fc3_super(spg_num)
        primitive_matrix = get_primitive_matrix(spg_num)

        ph3 = Phono3py(
            unitcell=unit_cell,
            supercell_matrix=fc3_super,
            phonon_supercell_matrix=fc2_super,
            symprec=1e-5,
        )

        if conf_fc3.get('cutoff', None) is not None:
            ph3.generate_displacements(
                distance=conf_fc3['displacement'],
                cutoff_pair_distance=conf_fc3['cutofff']
                )
        else:
            ph3.generate_displacements(
                distance=conf_fc3['displacement'],
                )

        if load_fc2:
            fc2 = ph_IO.parse_FORCE_CONSTANTS(f'{load_fc2}/FORCE_CONSTANTS_2ND_{idx}')
            ph3.fc2 = fc2
        else:
            try:
                ph3 = calculate_fc2(ph3, calc, symmetrize_fc2=symm_fc2)
                if save_fc2:
                    ph_IO.write_FORCE_CONSTANTS(
                        ph3.fc2,
                        filename=f'{save_fc2}/FORCE_CONSTANTS_2ND_{idx}',
                    )
            except Exception as e:
                sys.stderr.write(f'FC2 calc error at {idx}: {e}\n')
                FC2_Error = True

        write_csv(fc_logger, atoms, ph3, idx, spg_num, FC2_Error)
        ph3.save(f'{config["phonon"]["save"]}/phono3py_params_fc2_{idx}.yaml', settings={'compress': True})
        del atoms, ph3
        torch.cuda.empty_cache()
        gc.collect()
    fc_logger.close()
