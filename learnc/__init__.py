"""learnc: C programs whose `learned` variables are fitted to examples."""
from .program import Program, Learned
from .tune import tune, TuneResult
from .lift import candidates, lift
from .build import BuildError
from .machine import RunError

__version__ = "0.1.1"
