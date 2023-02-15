import configparser

config_fn = '/home/jovyan/.config/pycortex/options.cfg'

config = configparser.ConfigParser()
config.read(config_fn)
config['basic']['filestore'] = '/home/jovyan/work/DataLocal-w/pycortex_db_NSD'

with open(config_fn, 'w') as configfile:
    config.write(configfile)
    