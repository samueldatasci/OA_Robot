#---------------------------------------------------- Codigo Base Robot ----------------------------------------------------------------------------------------

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Initial, Stopping Station Positions, and Final Positions (Points in R^2)
e1, e2 = np.array([0, 0]), np.array([0.5, 0.5])  # Initial positions for Robots 1 and 2
s1, s2 = np.array([10, 2]), np.array([2, 10])  # Stopping stations for Robots 1 and 2
o1, o2 = np.array([7, 7]), np.array([8, 8])  # Final positions for Robots 1 and 2

# Particular instants where Robots should rest at the corresponding positions
T1, T2 = 15, 20  # Time at stopping stations constraints for Robots 1 and 2
Tf = 30  # Final time--when both Robots should rest at the final station

# Further Hyperparameters of the Problem
h = 1  # Sampling interval
beta = 1  # Dragging coefficient
m1, m2 = 2.0, 2.0  # Mass of the Robots
d = 4  # Maximum allowed communication distance (wireless limit)

# --------------------------Optimization Variables----------------------
R1 = cp.Variable((Tf, 2))  # Trajectory of Robot 1
R2 = cp.Variable((Tf, 2))  # Trajectory of Robot 2
V1 = cp.Variable((Tf, 2))  # Velocity over time of Robot 1
V2 = cp.Variable((Tf, 2))  # Velocity over time of Robot 2
F1 = cp.Variable((Tf-1, 2))  # Force applied to Robot 1 over time
F2 = cp.Variable((Tf-1, 2))  # Forces applied to Robot 2 over time

# ------------------------------Constraints---------------------------
constraints = []

# Initial conditions
constraints += [R1[0] == e1, R2[0] == e2]
constraints += [V1[0] == 0, V2[0] == 0]

# Final conditions
constraints += [R1[Tf-1] == o1, R2[Tf-1] == o2]
constraints += [V1[Tf-1] == 0, V2[Tf-1] == 0]

# Stopping station constraints
constraints += [R1[T1] == s1, R2[T2] == s2]

# Dynamics constraints due to Newton's law plus wireless communication constraint
for t in range(Tf - 1):
    constraints += [
        V1[t+1] == V1[t] + h * (F1[t] - beta * V1[t]) / m1,
        R1[t+1] == R1[t] + h * V1[t],
        V2[t+1] == V2[t] + h * (F2[t] - beta * V2[t]) / m2,
        R2[t+1] == R2[t] + h * V2[t],
        cp.norm(R1[t] - R2[t]) <= d  # Wireless communication constraint
    ]

# --------------------- objective function -----------------------------
gamm = 1 # Penalize large forces
lamb = 0 #Suggestion: should be set by the order of 100000. It penalizes large distances (Victoriya's credits)

objective = cp.Minimize(
    gamm * (cp.sum_squares(F1) + cp.sum_squares(F2))/((Tf-1)/h) +
    lamb * cp.sum_squares(R1[1:] - R1[:-1]) +  # Penalize travelled distance for Robot 1
    lamb * cp.sum_squares(R2[1:] - R2[:-1])    # Penalize travelled distance for Robot 2
)

#Solve the optimization problem: Minimize the objective function subject to the constraints
prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.SCS)

# Print solver status
print(f"Solver Status: {prob.status}")

# Check if solution was found before plotting
if prob.status not in ["infeasible", "unbounded"]:
    # Extract optimized positions
    opt_R1 = R1.value
    opt_R2 = R2.value

#--------------------- Plot 1: Optimized trajectories of the Robots ----------------------------------------

    plt.plot(opt_R1[:, 0], opt_R1[:, 1], 'bo-', label="Trajectory of Robot 1", zorder=1)
    plt.plot(opt_R2[:, 0], opt_R2[:, 1], 'ro-', label="Trajectory of Robot 2", zorder=1)
    plt.scatter([e1[0], e2[0]], [e1[1], e2[1]], c='black', marker='s', label="Initial Positions",
                zorder=3)
    plt.scatter([s1[0], s2[0]], [s1[1], s2[1]], c='green', marker='s', label="Stopping Stations",
                zorder=3)
    plt.scatter([o1[0], o2[0]], [o1[1], o2[1]], c='yellow', marker='s', label="Final Positions",
                zorder=3)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.title("Optimized Robot Trajectories")
    plt.grid()
    plt.show()


# --------------------- Plot 2: optimal forces deployed to the Robots --------------------------

    opt_F1 = F1.value
    opt_F2 = F2.value

    # Time steps
    time_steps = np.arange(Tf - 1)  # Forces are defined for Tf-1 steps

    # Extract optimal force components
    F1_x, F1_y = opt_F1[:, 0], opt_F1[:, 1]
    F2_x, F2_y = opt_F2[:, 0], opt_F2[:, 1]

    # Compute force magnitudes
    F1_magnitude = np.linalg.norm(opt_F1, axis=1)
    F2_magnitude = np.linalg.norm(opt_F2, axis=1)

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(time_steps, F1_x, 'b-', label="Force X (Robot 1)", linewidth=2)
    axs[0].plot(time_steps, F2_x, 'r-', label="Force X (Robot 2)", linewidth=2)
    axs[0].set_ylabel("Force X")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_steps, F1_y, 'b--', label="Force Y (Robot 1)", linewidth=2)
    axs[1].plot(time_steps, F2_y, 'r--', label="Force Y (Robot 2)", linewidth=2)
    axs[1].set_ylabel("Force Y")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(time_steps, F1_magnitude, 'b-', label="Force Magnitude (Robot 1)", linewidth=2)
    axs[2].plot(time_steps, F2_magnitude, 'r-', label="Force Magnitude (Robot 2)", linewidth=2)
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Force Magnitude")
    axs[2].legend()
    axs[2].grid()

    plt.suptitle("Force Components and Magnitudes Over Time")
    plt.show()

    print(f"The objective minimum is: {result}")
    sum_squares_F1 = np.sum(opt_F1 ** 2)  # Sum of F_x^2 + F_y^2 for Robot 1
    sum_squares_F2 = np.sum(opt_F2 ** 2)  # Sum of F_x^2 + F_y^2 for Robot 2
    total_sum_squares = (sum_squares_F1 + sum_squares_F2)/((Tf-1)*h)
    print(f"The minimum sum of square forces is: {total_sum_squares}")

else:
    print("Optimization problem is infeasible or unbounded.")
