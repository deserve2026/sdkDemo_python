import sys
sys.dont_write_bytecode = True
from _entry import deferred

if __name__ == '__main__':
    deferred('evaluate')

