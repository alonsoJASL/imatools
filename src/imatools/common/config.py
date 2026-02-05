import logging 

def configure_logging(log_name:str, log_level=logging.INFO, log_format='[%(funcName)s] %(message)s') :
    logger = logging.getLogger(log_name) 
    handler = logging.StreamHandler()
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger

def add_file_handler(logger, file_path: str) :
    file_handler = logging.FileHandler(file_path)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)