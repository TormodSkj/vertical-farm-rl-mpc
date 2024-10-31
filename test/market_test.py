import pytest
import numpy as np
from src.config import Config 
from src.market import Market
import matplotlib.pyplot as plt

def test_spot_price():
    
    N= 24*4
    grid = 'NO1'
    date = '2023-12-24'

    market = Market(N, 1133, grid, date)
    
    p_spot = market.get_spotprice()

    assert len(p_spot) == N

    
    plt.plot(np.linspace(0,N-1,N), p_spot)
    plt.ylabel("Spot price for grid: " + grid + " on " + date)
    plt.xlabel("Hour")
    plt.xticks(list(np.linspace(0,N,17)))

    # filename = "Spot_price_from_data"
    # foldername = "testing"
    # plt.savefig(config.plot_path + foldername + "/" + filename + ".png")    

    # assert plant.C_conv_PPFD == expected_value, f"Expected {expected_value}, but got {plant.C_conv_PPFD}"
    assert True