import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from market import Market
from plant import PlantModel

def test_c_conv_value():
    
    plant = PlantModel()
    # expected_value = 67.8125
    expected_value = 0.00027125
    assert plant.C_conv_PPFD == expected_value, f"Expected {expected_value}, but got {plant.C_conv_PPFD}"

def test_fw_calculation():

    #Sandbox to play around in mostly :D

    plant = PlantModel()
    X = np.ones((2, 30))
    print(plant.freshweight(X))

    assert True


def test_constants():

    plant = PlantModel(np.array([0,0]), 0)
    print(plant.C_conv_PPFD)
    print(plant.P_cap_max)
    print(plant.C_conv_PPFD*2.5)
    print(250*plant.C_conv_PPFD)

    assert False