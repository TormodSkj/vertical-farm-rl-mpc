import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from config import Config 
from market import Market
import matplotlib.pyplot as plt

def test_spot_price():
    
    N= 24*4
    grid = 'NO1'
    date = '2023-12-24'

    market = Market(N, 1133, grid, date)
    config = Config()
    
    p_spot = market.get_spotprice()

    assert len(p_spot) == N

    
    plt.plot(np.linspace(0,N-1,N), p_spot)
    plt.ylabel("Spot price for grid: " + grid + " on " + date)
    plt.xlabel("Hour")
    plt.xticks(list(np.linspace(0,N,17)))

    filename = "Spot_price_from_data"
    foldername = "testing"
    plt.savefig(config.plot_path + foldername + "/" + filename + ".png")    

    # assert plant.C_conv_PPFD == expected_value, f"Expected {expected_value}, but got {plant.C_conv_PPFD}"
    assert True


def test_Activation_probs():
    
    N= 24*4
    grid = 'NO1'
    date = '2023-12-24'

    config = Config()
    market = Market(N, 1133, grid, date)

    assert market.Pr_a_up(np.inf) == 0
    assert market.Pr_a_up(-np.inf) == 1
    assert market.Pr_a_dn(np.inf) == 0
    assert market.Pr_a_dn(-np.inf) == 1
    
    x = np.linspace(0,100,101)
    
    plt.plot(x, market.Pr_a_up(x), color = "blue",      label = "Probability of up activation")
    plt.plot(x, market.Pr_a_dn(x), color = "orange",    label = "Probability of down activation")
    plt.ylabel("Pr(A)")
    plt.xlabel("Price [Euro/MW]")
    plt.legend()

    filename = "Activation_probs"
    foldername = "testing"
    plt.savefig(config.plot_path + foldername + "/" + filename + ".png")    

    assert True