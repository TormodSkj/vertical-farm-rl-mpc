
import pytest

from src.plant import PlantModel 

def test_c_conv_value():
    
    plant = PlantModel()
    # expected_value = 67.8125
    expected_value = 0.00027125
    assert plant.C_conv_PPFD == expected_value, f"Expected {expected_value}, but got {plant.C_conv_PPFD}"