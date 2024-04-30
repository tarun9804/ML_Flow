import sys
import logging

logger = logging.getLogger(__name__)
def test():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0
    logger.info('inside test arg')
    print(alpha)
    print(sys.argv)
