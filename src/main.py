from controller import Controller
from plant import PlantModel
from market import Market
from utils import saveplot
from config import Config
import numpy as np
import matplotlib.pyplot as plt
import time

T = 7
N = 96*T    # 96 quarter hours per day
dt = 15*60  # 15 minutes times 60 seconds

Final_fw_sht = 80       #Final plant weight requirement

config = Config()
plant = PlantModel(Final_fw_sht)
market = Market(N, 1133, 'NO1', '2023-12-24')
controller = Controller(N, T, dt, plant, market)


start_time = time.time()
t, X_opt, U_opt, B_opt = controller.optimize()
end_time = time.time()

x_ts = np.array(X_opt)
u_ts = np.array(U_opt)[0,:]
b_ts = np.array(B_opt)

x1_ts = x_ts[0,1:]
x2_ts = x_ts[1,1:]

b_p_up = b_ts[0,:]
b_p_dn = b_ts[1,:]
b_c_up = b_ts[2,:]
b_c_dn = b_ts[3,:]

'''
plotting(t,[x1_ts[1:], x2_ts[1:]], "Combined_ocp_x")
plotting(t,[u_ts], "Combined_ocp_u")
plotting(t,[b_p_up, b_p_dn], "Combined_ocp_bp")
'''



foldername = "casadi_ocp"

##################################################
plt.figure(1)
plt.plot(t, x1_ts, "r", label="Structural dry weight (g/m^2)") 
plt.plot(t, x2_ts, "b", label="Non-structural dry weight (g/m^2)")
plt.plot(t, controller.model.freshweight(x_ts[:,1:]), "g", label="Freshweight shoot (g/plant)")
plt.axhline(y=controller.model.Final_fw_sht, color='orange', linestyle=':', label="Required Freshweight (g/plant)")
plt.ylabel("Weight")
plt.xlabel("Time (days)")
plt.legend()


filename = "Combined_ocp_x"
plt.savefig(config.plot_path + foldername + "/" + filename + ".png")

##################################################
plt.figure(2)
plt.plot(t, u_ts, label="U") 
plt.plot(t, controller.baseline_opt(), label="Baseline U") 
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
b_a_up = controller.market.Pr_a_up(b_c_up)
b_a_dn = controller.market.Pr_a_dn(b_c_dn)

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



#################################################
#Display the time it took running the program

elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)

print(f"OCP took: {int(minutes)} minutes and {seconds:.2f} seconds to solve.")