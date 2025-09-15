from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter, FrechetCellFilter
from ase.optimize import FIRE, BFGS

OPT_DICT = {'fire': FIRE, 'bfgs': BFGS}
FILTER_DICT = {'unitcell': UnitCellFilter, 'frechet': FrechetCellFilter}


class AseAtomRelax:
    def __init__(
        self,
        calc,
        opt,
        cell_filter=None,
        fix_symm=True,
        fmax=0.0001,
        steps=1000,
        log='-'
    ):
        self.calc = calc
        self.opt = opt
        self.cell_filter = cell_filter
        self.fix_symm = fix_symm
        self.fmax = fmax
        self.steps = steps
        self.log = log

    def relax_atoms(self, atoms):
        atoms = atoms.copy()
        atoms.calc = self.calc
        if self.fix_symm:
            atoms.set_constraint(FixSymmetry(atoms, symprec=1e-5))

        if self.cell_filter is not None:
            cf = self.cell_filter(atoms)
            opt = self.opt(cf, logfile=self.log)
        else:
            opt = self.opt(atoms, logfile=self.log)

        opt.run(fmax=self.fmax, steps=self.steps)
        steps = opt.get_number_of_steps()
        atoms = self.update_atoms(atoms)
        atoms.info['steps'] = steps
        atoms.info['opt'] = True
        if steps < self.steps:
            atoms.info['conv'] = True
        else:
            atoms.info['conv'] = False
        return atoms

    def update_atoms(self, atoms):
        atoms = atoms.copy()
        atoms.calc = self.calc
        atoms.info['e_fr_energy'] = atoms.get_potential_energy(force_consistent=True)
        atoms.info['e_0_energy'] = atoms.get_potential_energy()
        atoms.info['force'] = atoms.get_forces()
        atoms.info['volume'] = atoms.get_volume()
        atoms.info['a'] = atoms.cell.lengths()[0]
        atoms.info['b'] = atoms.cell.lengths()[1]
        atoms.info['c'] = atoms.cell.lengths()[2]
        atoms.info['alpha'] = atoms.cell.angles()[0]
        atoms.info['beta'] = atoms.cell.angles()[1]
        atoms.info['gamma'] = atoms.cell.angles()[2]
        return atoms


def get_ase_relaxer(conf, calc):
    arr_args = conf['args']
    arr_args.pop('relaxed_input_path', None)

    opt = OPT_DICT[arr_args['opt'].lower()]
    cell_filter = arr_args.get('cell_filter', None)
    if isinstance(cell_filter, str):
        cell_filter = FILTER_DICT[cell_filter.lower()]

    arr_args['calc'] = calc
    arr_args['opt'] = opt
    arr_args['cell_filter'] = cell_filter

    return AseAtomRelax(**arr_args)
    
    
