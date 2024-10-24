
from utils import *
from scipy.optimize import minimize
from acados_template import *



p_spot = generate_spotprice()
plotting(np.linspace(0, 23.75, 96), np.array([p_spot]), "spotprice")

# Example usage
x0 = np.array([1, 1])  # Initial state
# u_optimal = optimize_control(x0, p_spot)

# print("Optimized control inputs (u):", u_optimal)



def obj(x, x_init):
    cost = 0
    for i in range(len(x)):
        cost += x[i]**2
    
    return cost


def final_state_constraint(x):
    return x[-1] - 10

def dynamics_constraint(x):

    x_init = x[0]
    x_state = [x_init]
    u = x[6:]


    def x_dot(x, u):
        return -x + u

    for i in range(5):
        x_state.append(x_dot(x_state[i], u[i]))

    # return x_state


# Define the constraints for the optimizer
constr = {
    'type': 'eq',  # Equality constraint
    'fun': final_state_constraint,
    'args': ()  # Arguments for the constraint function
    ,
    'type': 'eq',  # Equality constraint
    'fun': dynamics_constraint,
    'args': ()  # Arguments for the constraint function
}


x0 = np.zeros(11)
result = minimize(obj, x0, args=(),
                      method='SLSQP', constraints= constr)

u_optimal = result.x

print(u_optimal)


