"""_summary_
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

# Imports from third party
import warnings

# suppress warnings
warnings.filterwarnings("ignore")

class Visualize:

    def __init__(self, training_instance):
        self.training_instance = training_instance