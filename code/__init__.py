"""
Information Fusion and Mining package for Python
==================================
IFM.py is a Python package integrating classical machine
learning, information fusion, and data mining algorithms
in the tightly-knit world of scientific Python packages
(sklearn, numpy, scipy, matplotlib).
See http://www.ifmlab.org/package.html for complete documentation.
"""

from importlib import util as _importlib_util
from pathlib import Path as _Path
import sysconfig as _sysconfig


# Torch imports pdb, and pdb expects Python's standard code helpers to exist.
# The project template also names this package code, so we expose those helpers.
_stdlib_code_path = _Path(_sysconfig.get_path('stdlib')) / 'code.py'
_stdlib_code_spec = _importlib_util.spec_from_file_location('_stdlib_code', _stdlib_code_path)
_stdlib_code = _importlib_util.module_from_spec(_stdlib_code_spec)
_stdlib_code_spec.loader.exec_module(_stdlib_code)

InteractiveInterpreter = _stdlib_code.InteractiveInterpreter
InteractiveConsole = _stdlib_code.InteractiveConsole
CommandCompiler = _stdlib_code.CommandCompiler
interact = _stdlib_code.interact
compile_command = _stdlib_code.compile_command
