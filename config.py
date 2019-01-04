import configparser

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
# config.read('./config_breakout.conf')
# config.read('./config_monsterkong_state.conf')
config.read('./config_monsterkong_image.conf')

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
curiosity_config = config['CURIOSITY']
mk_config = config["MONSTERKONG"]