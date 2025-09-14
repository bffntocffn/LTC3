from tqdm import tqdm
import gc
from ase.io import read, write

from ltc3.util.relax import get_ase_relaxer
from ltc3.util.phonopy_utils import get_spgnum


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

    csv_file.write(f'{idx},{opt},{atoms.info["formula"]},{sgnum},{atoms.info["e_fr_energy"]},{atoms.info["volume"]},{len(atoms)}{atoms.info["a"]},{atoms.info["b"]},{atoms.info["c"]},{atoms.info["alpha"]},{atoms.info["beta"]},{atoms.info["gamma"]},{conv},{steps}\n')


def relax_atoms_list(config, calc):
    atoms_list = read(config['data']['input_path'], **config['data']['input_args'])
    conf = config['relax']
    save_dir = conf['save']
    ase_relaxer = get_ase_relaxer(conf, calc)
    
    csv_log = open(f'{save_dir}/logger_logger.csv', 'a' buffering=1) if conf['cont'] else open(f'{save_dir}/relax.csv', 'w', buffering=1)
    csv_log.write('idx,opt,formula,sgn,energy,volume,natom,a,b,c,alpha,beta,gamma,conv,steps\n')

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
        write(f"{save_dir}/relaxed.extxyz", atoms, format='extxyz', append=True if idx > 0 else False)
        del atoms
        gc.collect()
