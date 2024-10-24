from controller import Controller
from plant import PlantModel
from market import Market
from utils import plotting
import numpy as np
import matplotlib.pyplot as plt

T = 7
N = 96*T
dt = 15*60 #15 minutes times 60 seconds


controller = Controller(N, T, dt)
# plant = PlantModel
# market = Market


t, X_opt, U_opt, B_opt = controller.optimize()

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

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, x1_ts, "r", label="Structural dry weight") 
plt.plot(t, x2_ts, "b", label="Non-structural dry weight")
plt.legend()  


filename = "Combined_ocp_x" + ".png"
plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
plt.savefig(plot_path)


plt.figure(2)
plt.plot(t, u_ts, label="PPFD") 
plt.legend()


filename = "Combined_ocp_u" + ".png"
plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
plt.savefig(plot_path)

plt.figure(3)
plt.plot(t, b_p_up, label="Bidding volume up") 
plt.plot(t, b_p_dn, label="Bidding volume down")
plt.legend()


filename = "Combined_ocp_b_p" + ".png"
plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
plt.savefig(plot_path)

plt.figure(4)
plt.plot(t, b_c_up, label="Bidding price up")
plt.plot(t, b_c_dn, label="Bidding price down")
plt.legend()


filename = "Combined_ocp_b_c" + ".png"
plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
plt.savefig(plot_path)



b_a_up = controller.market.Pr_a_up(b_c_up)
b_a_dn = controller.market.Pr_a_dn(b_c_dn)

plt.figure(5)
plt.plot(t, b_a_up, label="Probability of up-activation")
plt.plot(t, b_a_dn, label="Probability of down-activation")
plt.legend()

filename = "Combined_ocp_b_a" + ".png"
plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
plt.savefig(plot_path)


plt.figure(6)
plt.plot(t, controller.market.generate_spotprice(N), label="Spot price")
plt.legend()


filename = "Combined_ocp_p_spot" + ".png"
plot_path = "/home/tormodskj/vertical-farm-rl-mpc/plots/" + filename
plt.savefig(plot_path)
