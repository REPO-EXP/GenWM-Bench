import logging

from annotator.uniformer.mmcv.utils import get_logger

def get_root_logger(log_file=None, log_level=logging.INFO):
    
    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)

    return logger
