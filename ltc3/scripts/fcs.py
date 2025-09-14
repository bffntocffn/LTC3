import numpy as np
import os
import sys
from tqdm import tqdm

from ase import Atoms

from phono3py import Phono3py
from phono3py import file_IO as ph3_IO
from phonopy import file_IO as ph_IO

from ltc3.util.calc import SevenNetBatchCalculator, single_point_calculate_list
from ltc3.util.phonopy_utils import aseatoms2phonoatoms, get_primitive_matrix

def _get_fc2_supercell(atoms):
    sg_num = atoms.info['spg_num']
    if sg_num == 186:
        fc2_supercell = [5, 5, 3]

    elif sg_num in [216, 225]:
        fc2_supercell = [4, 4, 4]

    else:
        fc2_supercell = [3, 3, 3]
    return  fc2_supercell

def _get_fc3_supercell(atoms):
    sg_num = atoms.info['spg_num']
    if sg_num  == 186:
        fc3_supercell = [3, 3, 2]

    elif sg_num in [216, 225]:
        fc3_supercell = [2, 2, 2]

    else:
        fc3_supercell = [2, 2, 2]
    return fc3_supercell


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

    if isinstance(calc, SevenNetBatchCalculator):
        result = calc.batch_calculate(atoms_list, desc=desc)
    else:
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

    if isinstance(calc, SevenNetBatchCalculator):
        result = calc.batch_calculate(atoms_list, desc=desc)
    else:
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


def write_csv(csv_file, atoms, ph3, idx, FC2_Error, FC3_Error):
    # fc_logger.write(f'index,formula,spgnum,prim,fc2_super,fc3_super,fc2_disp,fc3_disp,fc2_error,fc3_error\n')
    fc2_supercell = _get_fc2_supercell(atoms)
    fc3_supercell = _get_fc3_supercell(atoms)
    prim_matrix = get_primitive_matrix(atoms)
    fc2_disp = len(ph3.phonon_supercells_with_displacements)
    fc3_disp = len(ph3.supercells_with_displacements)
    write(f'{idx},{atoms.info["formula"]},{atoms.info["spg_num"]},{prim_matrix},{fc2_super},{fc3_super},{fc2_disp},{fc3_disp},{FC2_Error},{FC3_Error}\n')

def process_fcs(config ,calc):
    conf_fc2, conf_fc3 = config['fc2'], config['fc3']
    save_fc2, save_fc3 = conf_fc2['save'], conf_fc3['save']
    symm_fc2, symm_fc3 = conf_fc2['symm'], conf_fc3['symm']
    load_fc2, load_fc3 = conf_fc2['load'], conf_fc3['load']

    atoms = read(f'{config["relax"]["save"]}/relaxed.extxyz', **config['data']['input_args']) 
    fc_logger = open(f'{config["phonon"]["save"]}/fc_logger.csv', 'w', buffering=1)
    fc_logger.write(f'index,formula,spgnum,prim,fc2_super,fc3_super,fc2_disp,fc3_disp,fc2_error,fc3_error\n')

    for idx, atoms in enumerate(tqdm(atoms_list, desc='processing fcs')):
        FC2_Error, FC3_Error = False, False
        unit_cell = aseatoms2phonoatoms(atoms)
        fc2_supercell = _get_fc2_supercell(atoms)
        fc3_supercell = _get_fc3_supercell(atoms)
        primitive_matrix = get_primitive_matrix(atoms)

        ph3 = Phono3py(
            unitcell=unit_cell,
            supercell_matrix=fc3_supercell,
            phonon_supercell_matrix=fc2_supercell,
            symprec=1e-5,
        )

        if conf_fc3.get('cutoff', None) is not None:
            ph3.generate_displacements(
                distance=conf['displacement'],
                cutoff_pair_distance=conf_fc3['cutofff']
                )
        else:
            ph3.generate_displacements(
                distance=conf['displacement'],
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

        if load_fc3:
            fc3 = ph3_IO.read_fc3_from_hdf5(f'{load_fc3}/fc3_{idx}.hdf5')
            ph3.fc3 = fc3
        else:
            try:
                ph3 = calculate_fc3_phono3py(ph3, calc, symmetrize_fc3=symm_fc3)
                if save_fc3:
                    ph3_IO.write_fc3_to_hdf5(
                        ph3.fc3,
                        filename=f'{save_fc3}/fc3_{idx}.hdf5',
                    )
            except Exception as e:
                sys.stderr.write(f'FC3 calc error at {idx}: {e}\n')
                FC3_Error = True

        write_csv(fc_logger, atoms, ph3, idx, FC2_Error, FC3_Error)
        ph3.save(f'{config["phonon"]["save"]}/phono3py_params_{idx}.yaml', settings={'compress': True})
        del atoms, ph3
        gc.collect()
