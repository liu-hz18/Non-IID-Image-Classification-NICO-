import os
import logging
from configparser import ConfigParser


class Logger(object):

    def __init__(self, name: str, directory: str):
        # CRITICAL > ERROR > WARNING > INFO > DEBUG
        self.dir = directory
        file_name = os.path.join(directory, "log.txt")
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s]  %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(file_name, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        self._logger = logger

    def info(self, message: str):
        self._logger.info(message)

    def debug(self, message: str):
        self._logger.debug(message)

    def warn(self, message: str):
        self._logger.warn(message)

    def save_args(self, args):
        config = ConfigParser()
        config['configs'] = vars(args)
        with open(os.path.join(self.dir, "config.txt"), 'w', encoding='utf-8') as f:
            config.write(f)
