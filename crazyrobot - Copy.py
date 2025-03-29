
def forcecharts(force1, mass1, h, beta):

#--------------------- Plot 1: Optimized trajectories of the Robots ----------------------------------------

    plt.plot(opt_R1[:, 0], opt_R1[:, 1], 'bo-', label="Trajectory of Robot 1", zorder=1)
    plt.scatter([initpos1[0]], [initpos1[1]], c='black', marker='s', label="Initial Positions",
                zorder=3)
    plt.scatter([stoppos1[0]], [stoppos1[1]], c='green', marker='s', label="Stopping Stations",
                zorder=3)
    plt.scatter([finalpos1[0]], [finalpos1[1]], c='yellow', marker='s', label="Final Positions",
                zorder=3)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.title("Robot Trajectory, optimized for minimum force")
    plt.grid()
    plt.show()



# --------------------- Plot 2: optimal forces deployed to the Robots --------------------------

    opt_F1 = force1.value

    # Time steps
    #time_steps = np.arange(Tf - 1)  # Forces are defined for Tf-1 steps
    time_steps1 = np.arange(finaltime1)  # Forces are defined for finaltime1-1 steps

    # Extract optimal force components
    F1_x, F1_y = opt_F1[:, 0], opt_F1[:, 1]

    # Compute force magnitudes
    F1_magnitude = np.linalg.norm(opt_F1, axis=1)

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(time_steps1, F1_x, 'b-', label="Force X (Robot 1)", linewidth=2)
    axs[0].set_ylabel("Force X")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_steps1, F1_y, 'b--', label="Force Y (Robot 1)", linewidth=2)
    axs[1].set_ylabel("Force Y")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(time_steps1, F1_magnitude, 'b-', label="Force Magnitude (Robot 1)", linewidth=2)
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Force Magnitude")
    axs[2].legend()
    axs[2].grid()

    plt.suptitle("Force Components and Magnitudes Over Time")
    plt.show()







def print_results(result, force1, mass1, h, beta):
    opt_F1 = force1.value

    print(f"**** Sampling interval h={h}, inertia beta = {beta}")
    print(f"*** Optimal Forces for Robot 1: {opt_F1[:2]} ...")
    
    indivforces = [np.linalg.norm(forceind) for forceind in opt_F1]
    #indivforces = [np.linalg.norm(opt_F1[i]) for i in range(len(opt_F1))]
    print(f"*** Individual forces: {indivforces[:2]} ...")

#     #print(f"*** Newcalc {sum([cp.sum_squares(force1[t]) for t in range(finaltime1)])}")
#     print(f"*** Newcalx {np.sum([cp.norm(tt)**2/h for tt in opt_F1])}")
# #    cp.sum([cp.sum_squares(force1[t]) / mass1[t] for t in range(finaltime1)])

    
#     totforc = sum(indivforces)
#     print(f"*** Total force: {totforc}")


    # ## print(f"The objective minimum is: {result:.0f}")
    # sum_squares_F1 = np.sum(opt_F1 ** 2)  # Sum of F_x^2 + F_y^2 for Robot 1
    
    # print(f"*** Sum of squares of forces: {sum_squares_F1[0:2]}")
    # energy_consumption = np.sum((np.linalg.norm(opt_F1, axis=1)**2 / np.array(mass1)) * h)

    # #total_sum_squares = (sum_squares_F1 + sum_squares_F2)/((Tf-1)/h)
    # total_sum_squares = (sum_squares_F1)/((finaltime1-1)*h)
    # print(f"The minimum sum of square forces is: {total_sum_squares}")
    # print(f"Energy: {energy_consumption}")


    # print("Norm of F1 at each time step:")
    # print(f"np.linalg.norm(opt_F1, axis=1): {np.linalg.norm(opt_F1[:3], axis=1)}")
    # print("Squared norml of F1 at each time step:")
    # print(f"np.linalg.norm(opt_F1, axis=1) ** 2: {np.linalg.norm(opt_F1[:3], axis=1) ** 2}")
    # #print(f"Mass1: {mass1}")
    # #print(f"position1: {opt_R1}")

    # sum_squares_F1 = np.sum(np.linalg.norm(opt_F1, axis=1) ** 2 * h)  # Correct squared sum
    # total_energy = (sum_squares_F1)  # Corrected energy computation
    # totenerg2 = cp.sum( cp.sum_squares(force1) / mass1)
    # totenerg3 = cp.sum( cp.sum_squares(opt_F1) ** 2 / mass1)
    # totenerg4 = np.sum(np.linalg.norm(opt_F1, axis=1) ** 2)
    # print(f"Total energy consumption: {total_energy}")
    # print(f"Total energy consumption2: {totenerg2}")
    # print(f"Total energy consumption3: {totenerg3}")
    # print(f"Total energy consumption4: {totenerg4}")

    # print("Sum of squares of forces:")
    # print(cp.sum_squares(force1))
    # print(cp.sum_squares(opt_F1))




import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt



# Initial, Stopping Station Positions, and Final Positions (Points in R^2)
initpos1 = np.array([0, 0])  # Initial position for the Robot
stoppos1 = np.array([10, 2])  # Stopping station for the Robot
finalpos1 = np.array([7, 7])  # Final position for the Robot

# Particular instant where Robot should rest at the corresponding position
stoptime1 = 20  # Time at stopping station constraint for Robot
finaltime1 = 30  # Final time--when the Robos should rest at the final station

# Further Hyperparameters of the Problem
h = 0.01  # Sampling interval
beta = 0.0  # Dragging coefficient was = 1
massrobot1 = 2.0  # Mass of the Robot
masspackage1 = 0.0  # Mass of the Package dropped at the stop station

mass1 = [massrobot1 + masspackage1 if i <= stoptime1 else massrobot1 for i in range(finaltime1)]


# --------------------------Optimization Variables----------------------
path1 = cp.Variable((finaltime1, 2))  # Trajectory of Robot 1
velocity1 = cp.Variable((finaltime1, 2))  # Velocity over time of Robot 1

#SS# force1 = cp.Variable((finaltime1-1, 2))  # Force applied to Robot 1 over time
force1 = cp.Variable((finaltime1, 2))  # Force applied to Robot 1 over time


# ------------------------------Constraints---------------------------
constraints = []

# Initial conditions
constraints += [path1[0] == initpos1]
constraints += [velocity1[0] == 0]

# Final conditions
constraints += [path1[finaltime1-1] == finalpos1]
constraints += [velocity1[finaltime1-1] == 0]

# Stopping station constraints
constraints += [path1[stoptime1] == stoppos1]
#constraints += [velocity1[stoptime1-1] == 0]

print(constraints.count)
# Dynamics constraints due to Newton's law
for t in range(finaltime1 - 1):
    constraints += [
        velocity1[t+1] == velocity1[t] + h * (force1[t] - beta * velocity1[t]) / mass1[t],
        path1[t+1] == path1[t] + h * velocity1[t]
    ]

print(constraints.count)


# --------------------- objective function -----------------------------
objective = cp.Minimize(

    #cp.sum( cp.sum_squares(force1) / mass1)
    #### cp.sum( cp.sum_squares(force1) / mass1)
	#cp.sum(cp.multiply([1 for x in mass1] / mass1, cp.sum_squares(force1)))
#	cp.sum(cp.multiply([1/x for x in mass1], cp.sum_squares(force1)))

    cp.sum([cp.sum_squares(force1[t]) / mass1[t] for t in range(finaltime1)])

    )

#Solve the optimization problem: Minimize the objective function subject to the constraints
prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.SCS)

# Print solver status
print(f"Solver Status: {prob.status}")

# Check if solution was found before plotting
if prob.status not in ["infeasible", "unbounded"]:
    # Extract optimized positions
    opt_R1 = path1.value

    print_results(result, force1, mass1, h, beta)

else:
    print("Optimization problem is infeasible or unbounded.")

