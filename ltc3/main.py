import os
import sys
from tqdm import tqdm
import warnings
import yaml

from ase.io import read, write

from ltc3.util.calc import calc_from_config
from ltc3.util.relax import get_ase_relaxer
from ltc3.util.utils import dumpYAML
from ltc3.scripts.parse_input import parse_config
from ltc3.scripts.relax import process_relax
from ltc3.scripts.fc2 import process_fc2
from ltc3.scripts.fc3 import process_fc3
from ltc3.scripts.conductivity import process_conductivity


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dumpYAML(config, filename='config_parsed.yaml')

    config = parse_config(config)
    calc = calc_from_config(config)
    if config['relax']['run']:
        process_relax(config, calc)

    if config['fc2']['run']:
        process_fc2(config, calc)

    if config['fc3']['run']:
        process_fc3(config, calc)

    if config['cond']['run']:
        process_conductivity(config)

if __name__=='__main__':
    main()
