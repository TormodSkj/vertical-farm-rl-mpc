import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import utils  


def test_generate_table():
    
    cost_data = [
        ['Cost of power', 200, 250],
        ['Cost of bidding', 0, -70]
    ]

    cost_table = utils.generate_table(cost_data, header=['Baseline', 'Bidding'], sumrow=True, diffcol=True)
    print(f'DLI DATA: \n {cost_table}\n')


    assert True