import os

class Essential:
    pass


DEFAULT_DATA_CONFIG = {
    'input': Essential(),
    'input_args': {'index': ':'},
}


DEFAULT_CALC_CONFIG = {
    'calc_type': 'sevennet',
    'path': Essential(),
    'calc_args': {},
    'batch_size': None,
    'avg_atom_num': None,
}


DEFAULT_RELAX_CONFIG = {
    'save': Essential(),
    'load': None,
    'cont': False,
    'args': {},
}


DEFAULT_PHONON_CONFIG = {
    'save': None
    }

DEFAULT_FC2_CONFIG = {
    'save': None,
    'load': None,
    'symm': True

}

DEFAULT_FC3_CONFIG = {
    'save': None,
    'load': None,
    'symm': True,
    'displacement': None,
    'cutoff': None

}

DEFAULT_COND_CONFIG = {
    'save': None,
    'cond_type': 'bte',
    'temperature': 300,
    'is_isotope': True,
    'is_LBTE': True
}


def update_config_with_defaults(config):
    key_parse_pair = {
        'data': DEFAULT_DATA_CONFIG,
        'calculator': DEFAULT_CALC_CONFIG,
        'relax': DEFAULT_RELAX_CONFIG,
        'phonon': DEFAULT_PHONON_CONFIG,
        'fc2': DEFAULT_FC2_CONFIG,
        'fc3': DEFAULT_FC3_CONFIG,
        'cond': DEFAULT_COND_CONFIG,
    }

    for key, default_config in key_parse_pair.items():
        config_parse = default_config.copy()
        config_parse.update(config[key])

        for k, v in config_parse.items():
            if not isinstance(v, Essential):
                continue
            raise ValueError(f'{key}: {k} must be given')
        config[key] = config_parse

    return config


def _isinstance_in_list(inp, insts):
    return any([isinstance(inp, inst) for inst in insts])


def _islistinstance(inps, insts):
    return all([_isinstance_in_list(inp, insts) for inp in inps])


def check_calc_config(config):
    conf = config['calculator']
    assert conf['calc_type'].lower() in ['sevennet', 'sevennet-batch', 'custom']
    assert isinstance(conf['path'], str)
    assert _isinstance_in_list(conf['batch_size'], [int, type(None)])
    assert _isinstance_in_list(conf['avg_atom_num'], [int, type(None)])


def check_relax_config(config):
    conf = config['relax']
    if (load := conf['load']) is not None:
        assert os.path.isfile(load)
        return

    assert os.path.isfile(config['data']['input'])


def check_phonon_config(config):
    conf = config['phonon']
    os.makedirs(conf['save'], exist_ok=True)


def check_fc2_config(config):
    conf = config['fc2']
    os.makedirs(conf['save'], exist_ok=True)
    if (load_fc2 := conf['load']) is not None:
        assert os.path.isdir(load_fc2)
    assert isinstance(conf['symm'], bool)


def check_fc3_config(config):
    conf = config['fc3']
    os.makedirs(conf['save'], exist_ok=True)
    if (load_fc3 := conf['load']) is not None:
        assert os.path.isdir(load_fc3)
    assert isinstance(conf['symm'], bool)
    assert isinstance(conf['displacement'], float)


def check_cond_config(config):
    conf = config['cond']
    os.makedirs(conf['save'], exist_ok=True)
    assert conf['cond_type'] in ['bte', 'wte']
    assert (
        _isinstance_in_list(conf['temperature'], [float, int])
        or _islistinstance(conf['temperature'], [float, int])
    )
    assert isinstance(conf['is_isotope'], bool)


def parse_config(config):
    config = update_config_with_defaults(config)
    check_calc_config(config)
    check_relax_config(config)
    check_phonon_config(config)
    check_fc2_config(config)
    check_fc3_config(config)
    check_cond_config(config)

    return config
