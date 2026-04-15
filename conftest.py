"""
To resolve pytest errors from files with same name 
(ex. loss/classification.py and metrics/classification.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))