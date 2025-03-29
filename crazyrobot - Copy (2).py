import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt



def forcecharts(positions, times, path1, force1): #, mass1, h, beta):
    """    Function to plot the trajectory of the robots and the forces applied to them.
    Args:
        positions (list): List of positions for the robots.
        times (list): List of times for stopping and final positions.
        path1 (cvxpy.Variable): Optimized trajectory of Robot 1.
        force1 (cvxpy.Variable): Optimized forces applied to Robot 1.
    """
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 9), sharex=False)

#--------------------- Plot 1: Optimized trajectories of the Robots ----------------------------------------
    opt_R1 = path1.value


    axs[0].plot(opt_R1[:, 0], opt_R1[:, 1], 'bo-', label="Trajectory of Robot 1", zorder=1)
    axs[0].scatter([positions[0][0]], [positions[0][1]], c='green', marker='s', label="Initial Positions",
                zorder=3)
    axs[0].scatter([positions[-1][0]], [positions[-1][1]], c='red', marker='s', label="Final Positions",
                zorder=3)
    # for pos in positions[1:-1]:
    #     axs[0].scatter([pos[0]], [pos[1]], c='yellow', marker='o', label="Stopping Stations", zorder=2)
    axs[0].scatter([pos[0] for pos in positions[1:-1]],
                [pos[1] for pos in positions[1:-1]],
                c='yellow', marker='o', label="Stopping Stations", zorder=2)

    axs[0].set_xlabel("X Position")
    axs[0].set_ylabel("Y Position")
    axs[0].legend()
    axs[0].set_title = "Robot Trajectory, optimized for minimum force"
    axs[0].grid()
    
    #plt.show()



# --------------------- Plot 2: optimal forces deployed to the Robots --------------------------

    opt_F1 = force1.value

    # Time steps
    time_steps1 = np.arange(len(path1.value))

    # Extract optimal force components
    F1_x, F1_y = opt_F1[:, 0], opt_F1[:, 1]

    # Compute force magnitudes
    F1_magnitude = np.linalg.norm(opt_F1, axis=1)


    axs[1].plot(time_steps1, F1_magnitude, 'b-', label="Force Magnitude (Robot 1)", linewidth=2)
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Force Magnitude")
    axs[1].set_title("Force Magnitudes Over Time")
    if len(positions) > 2:

        axs[1].axvline(x=times[0], color='yellow', linestyle='--', label="Stopping Station", zorder=2)

    axs[1].legend()
    axs[1].grid()

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




#1. Allow for n intermediate stopping stations (also adjust times and weights accordingly)

def optimize_robots(positions = [[0,0], [10,2], [7,7]],
                    times = [20, 30],
                    h = 0.01, # sampling interval
                    beta = 0.0, # dragging coefficient (was = 1)
                    massrobot1 = 2.0,
                    masspackage1 = 0.0,
                    stoptodrop=True):
    """    Function to optimize the trajectory of robots based on given parameters.
    Args:
        positions (list): List of positions for the robots.
        times (list): List of times for stopping and final positions.
        h (float): Sampling interval.
        beta (float): Dragging coefficient.
        massrobot1 (float): Mass of the robot.
        masspackage1 (float): Mass of the package dropped at the stop station."
        """
    # Initial, Stopping Station Positions, and Final Positions (Points in R^2)
    initpos1 = np.array(positions[0])  # Initial position for the Robot
    if len(positions) > 2:
        stoppos1 = np.array(positions[1])
    else:
        stoppos1 = np.nan
    finalpos1 = np.array(positions[-1])  # Final position for the Robot

    # Particular instant where Robot should rest at the corresponding position
    if len(times) > 1:
        stoptime1 = times[0]  # Time at stopping station constraint for Robot
    else:
        stoptime1 = np.nan
    finaltime1 = times[-1]  # Final time--when the Robos should rest at the final station

    if len(positions) > 2:
        mass1 = [massrobot1 + masspackage1 if i <= stoptime1 else massrobot1 for i in range(finaltime1)]
    else:
        mass1 = [massrobot1+masspackage1 for i in range(finaltime1)]


    # --------------------------Optimization Variables----------------------
    path1 = cp.Variable((finaltime1, 2))  # Trajectory of Robot 1
    velocity1 = cp.Variable((finaltime1, 2))  # Velocity over time of Robot 1
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
    if len(positions) > 2:
        constraints += [path1[stoptime1] == stoppos1]
        if stoptodrop:
            constraints += [velocity1[stoptime1] == 0]
    else:
        pass
        # If no stopping station, set the path at the final time to be the same as the final position
        # This is a workaround for the case where there are no stopping stations
        #constraints += [path1[stoptime1] == stoppos1]

    # Dynamics constraints due to Newton's law
    for t in range(finaltime1 - 1):
        constraints += [
            velocity1[t+1] == velocity1[t] + h * (force1[t] - beta * velocity1[t]) / mass1[t],
            path1[t+1] == path1[t] + h * velocity1[t]
        ]

   
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
    print("Problem:")
    #print(prob)
    print(f"Solver Status: {prob.status}")

    # Check if solution was found before plotting
    if prob.status not in ["infeasible", "unbounded"]:
        # Extract optimized positions
        #opt_R1 = path1.value

        print_results(result, force1, mass1, h, beta)
        forcecharts(positions, times, path1, force1)

    else:
        print("Optimization problem is infeasible or unbounded.")



for h in [0.01]:
    optimize_robots(stoptodrop=False, h=h, positions=[[0,0], [3,3] [10,2],[9,8], [7,7]], times=[5, 20, 24, 30])
    #optimize_robots(stoptodrop=False, h=h)
# optimize_robots(stoptodrop=True)

