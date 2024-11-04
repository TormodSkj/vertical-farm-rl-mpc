from controller import Controller
from plant import PlantModel
from market import Market
from utils import saveplot
from config import Config
import numpy as np
import matplotlib.pyplot as plt
import time

T = 7
N = 96*T                    # 96 quarter hours per day
dt = 15*60                  # 15 minutes times 60 seconds

x_init = np.array([5, 1])   # Specify init vector
Final_fw_sht = 80           # Final plant weight requirement

config = Config()
plant = PlantModel(x_init, Final_fw_sht)
market = Market(N, 1133, 'NO1', '2023-12-24')
controller = Controller(N, T, dt, plant, market, "opt")     #Baseline: 'opt' / 'rigid'


#################################################
start_time = time.time()
controller.optimize()
end_time = time.time()

elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)

print(f"OCP took: {int(minutes)} minutes and {seconds:.2f} seconds to solve.")
#################################################


t = controller.t
u_opt = controller.u_opt
u_base = controller.u_base

x_opt = controller.x_opt
x1_opt = x_opt[0,1:]
x2_opt = x_opt[1,1:]
x_base = controller.x_base
x1_base = x_base[0,1:]
x2_base = x_base[1,1:]

b_p_up = controller.B_opt[0,:]
b_p_dn = controller.B_opt[1,:]
b_c_up = controller.B_opt[2,:]
b_c_dn = controller.B_opt[3,:]
b_a_up = controller.market.Pr_a_up(b_c_up)
b_a_dn = controller.market.Pr_a_dn(b_c_dn)

eps = controller.Eps_opt

print(f"Missing fresh weight: {eps}g per plant")

'''
plotting(t,[x1_ts[1:], x2_ts[1:]], "Combined_ocp_x")
plotting(t,[u_ts], "Combined_ocp_u")
plotting(t,[b_p_up, b_p_dn], "Combined_ocp_bp")
'''

bidding_earnings_up = np.multiply(b_a_up, b_p_up, b_c_up)/4
bidding_earnings_dn = np.multiply(b_a_dn, b_p_dn, b_c_dn)/4
f_opt = (np.sum(bidding_earnings_up) + np.sum(bidding_earnings_dn))
f_base = controller.f_base

print(f"\nCost of base: {f_base}")
print(f"Cost after bidding: {f_base - f_opt}")
print(f"Cost reduction from bidding: {f_opt}")
print(f"Reduction in percentage: {100*f_opt/(f_base - f_opt)} \n")


#################################################
#                   PLOTTING                    #
#################################################

foldername = "casadi_ocp"

##################################################
plt.figure(1)
plt.plot(t, x1_opt, "r", label="Structural dry weight (g/m^2)") 
plt.plot(t, x2_opt, "b", label="Non-structural dry weight (g/m^2)")
plt.plot(t, controller.model.freshweight(x_opt[:,1:]), "g", label="Freshweight shoot (g/plant)")
plt.plot(t, controller.model.freshweight(x_base[:,1:]), color='purple', linestyle=':', label="Baseline freshweight shoot (g/plant)")
plt.axhline(y=controller.model.Final_fw_sht, color='orange', linestyle=':', label="Required Freshweight (g/plant)")
plt.ylabel("Weight")
plt.xlabel("Time (days)")
plt.legend()


filename = "Combined_ocp_x"
plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

##################################################
plt.figure(2)
plt.plot(t, u_opt, label="U") 
plt.plot(t, u_base, label="Baseline U") 
plt.ylabel("Light level (PPFD)")
plt.xlabel("Time (days)")
plt.legend()


filename = "Combined_ocp_u"
plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

##################################################
plt.figure(3)
plt.plot(t, b_p_up, label="Bidding volume up") 
plt.plot(t, b_p_dn, label="Bidding volume down")
plt.ylabel("Bidding volume (MW)")
plt.xlabel("Time (days)")
plt.legend()

filename = "Combined_ocp_b_p"
plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

##################################################
plt.figure(4)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Bidding price up
ax1.plot(t, b_c_up, label="Bidding price up", color="blue")
ax1.set_ylabel("Bidding Price Up")
ax1.legend(loc="upper right")

# Bidding price down
ax2.plot(t, b_c_dn, label="Bidding price down", color="red")
ax2.plot(t, market.get_spotprice(), label="Spot price", color="blue")
ax2.set_ylabel("Bidding Price Down")
ax2.set_xlabel("Time")
ax2.legend(loc="upper right")

fig.tight_layout()
filename = "Combined_ocp_b_c"
plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

##################################################
plt.figure(6)
plt.plot(t, b_a_up, label="Up-activation")
plt.plot(t, b_a_dn, label="Down-activation")
plt.ylabel("Probability of activations")
plt.xlabel("Time (days)")
plt.legend()

filename = "Combined_ocp_b_a"
plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

##################################################
plt.figure(7)
plt.plot(t, market.get_spotprice(), label="Spot price")
plt.ylabel("Spot price (kr/kWh)")
plt.xlabel("Time (days)")
plt.legend()


filename = "Combined_ocp_p_spot"
plt.savefig(config.plot_path + foldername + "/" + filename + ".png")



