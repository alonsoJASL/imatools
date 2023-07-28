import logging 

def configure_logging(log_name:str, log_level=logging.INFO, log_format='[%(funcName)s] %(message)s') :
    logger = logging.getLogger(log_name) 
    handler = logging.StreamHandler()
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger
