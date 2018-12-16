import configparser

config = configparser.ConfigParser()
# config.read('./config_breakout.conf')
config.read('./config_monsterkong.conf')

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
default_config = config[default]
mk_config = config["MONSTERKONG"]