import os
import sys
from tqdm import tqdm
import warnings
import yaml

from ase.io import read, write

from ltc3.util.calc import calc_from_config
from ltc3.util.relax import get_ase_relaxer
from ltc3.scripts.parse_input import parse_config
from ltc3.scripts.relax import process_relax
from ltc3.scripts.fcs import process_fcs
from ltc3.scripts.conductivity import process_conductivity


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = parse_config(config)
    calc = calc_from_config(config)
    process_relax(config, calc)
    process_fcs(config, calc)

if __name__=='__main__':
    main()
