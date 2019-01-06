import configparser
import os

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
# config.read('./config_breakout.conf')
# config.read('./config_monsterkong_state.conf')
config_file = os.environ.get('CONFIG_FILE')
if not config_file:
    raise ValueError("You must set the CONFIG_FILE variable")

print("Using config file: " + str(config_file))
config.read(config_file)

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
curiosity_config = config['CURIOSITY']
mk_config = config["MONSTERKONG"]