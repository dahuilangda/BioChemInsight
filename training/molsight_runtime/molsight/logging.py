import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)


def get_logger(name: str = None) -> logging.Logger:
    r"""
    Returns a logger with the specified name. It it not supposed to be accessed externally.
    """
    if name is None:
        name = __name__
    return logging.getLogger(name)
