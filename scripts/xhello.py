#!/usr/bin/env python
import sys
import sh
print(sh.mpiexec(['-n', '3', sys.executable, 'hello.py']))
