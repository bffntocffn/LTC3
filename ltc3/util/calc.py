import numpy as np
from tqdm import tqdm

from ase.calculators.singlepoint import SinglePointCalculator
from sevenn.calculator import SevenNetCalculator


def calc_from_config(config):
    calc_config = config['calculator']
    calc_type = calc_config['calc_type'].lower()
    calc_args = calc_config.get('calc_args', {})
    return SevenNetCalculator(model=calc_config['path'], **calc_args)



def single_point_calculate(atoms, calc):
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    calc_results = {"energy": energy, "forces": forces, "stress": stress}
    calculator = SinglePointCalculator(atoms, **calc_results)
    new_atoms = calculator.get_atoms()

    return new_atoms


def single_point_calculate_list(atoms_list, calc, desc=None):
    calculated = []
    for atoms in tqdm(atoms_list, desc=desc, leave=False):
        calculated.append(single_point_calculate(atoms, calc))

    return calculated
