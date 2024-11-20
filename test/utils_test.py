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



def test_get_metrics_table():
    
    runs = {
        "Baseline": {
            "metrics": {
                "elapsed_time": 24.320860385894775,
                "f": 178746.88639397718,
                "eps": -9.97489847166543e-09,
                "DLI_avg": 12.757685370393883,
                "DLI_max": 20.60023685046,
                "DLI_min": 0.8932590116359744,
                "Costs": 178746.89636887563,
                "Earnings": 0,
                "Total": 178746.89636887563
            },
            'timeseries' : {}
        },
        'Bidding':{
            "metrics": {
                "elapsed_time": 178.25243473052979,
                "f": 146709.01275754368,
                "eps": -9.974898534898046e-09,
                "DLI_avg": 12.602382671661776,
                "DLI_max": 20.892361617490153,
                "DLI_min": 2.312984872926231,
                "Costs": 146709.02273244222,
                "Earnings": 29680.466302413446,
                "Total": 117028.55643002877
                },
            'timeseries' : {}
        }
    }

    cost_table = utils.get_metrics_table(runs)
    print(f'DLI DATA: \n {cost_table}\n')

    assert True

