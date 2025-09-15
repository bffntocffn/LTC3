from tqdm import tqdm
import gc
import ase.io as ase_IO

from ltc3.util.relax import get_ase_relaxer
from ltc3.util.phonopy_utils import get_spgnum

import pickle


def write_csv(csv_file, atoms, idx):
    try:
        opt = atoms.info['opt']
        conv = atoms.info['conv']
        steps = atoms.info['steps']
        sgnum = atoms.info["spg_num"]
    except:
        opt = 'pre'
        conv = 'na'
        steps = 'na'
        sgnum = atoms.info["init_spg_num"]
    formula = atoms.get_chemical_formula(empirical=True)

    csv_file.write(f'{idx},{opt},{formula},{sgnum},{atoms.info["e_fr_energy"]},{atoms.info["volume"]},{len(atoms)}{atoms.info["a"]},{atoms.info["b"]},{atoms.info["c"]},{atoms.info["alpha"]},{atoms.info["beta"]},{atoms.info["gamma"]},{conv},{steps}\n')


def process_relax(config, calc):
    atoms_list = ase_IO.read(config['data']['input'], **config['data']['input_args'])
    conf = config['relax']
    save_dir = conf['save']
    ase_relaxer = get_ase_relaxer(conf, calc)
    
    csv_log = open(f'./relax_logger.csv', 'a', buffering=1) if conf['cont'] else open(f'./relax_logger.csv', 'w', buffering=1)
    csv_log.write('idx,opt,formula,sgn,energy,volume,natom,a,b,c,alpha,beta,gamma,conv,steps\n')

    relax_dct = {}
    relax_dct['calculator'] = config['calculator']['path']
    relax_dct['modal'] = config['calculator']['calc_args']['modal']

    for idx, atoms in enumerate(tqdm(atoms_list, desc='relaxing atoms')):

        atoms.info['idx'] = idx
        atoms.info['formula'] = atoms.get_chemical_formula(empirical=True)
        atoms.info['opt'] = 'pre'
        atoms.info['init_spg_num'] = init_spg = get_spgnum(atoms)
        atoms = ase_relaxer.update_atoms(atoms)
        write_csv(csv_log, atoms, idx)

        atoms = ase_relaxer.relax_atoms(atoms)
        atoms = ase_relaxer.update_atoms(atoms)
        spg_num = atoms.info['spg_num'] = get_spgnum(atoms)
        write_csv(csv_log, atoms, idx)
        atoms.calc = None
        ase_IO.write(f'{save_dir}/CONTCAR_{idx}', atoms, format='vasp')
        relax_dct[idx] = atoms.info.copy()
    csv_log.close()
    with open(f'{save_dir}/relax_dct.pkl', 'wb') as f:
        pickle.dump(relax_dct, f)
