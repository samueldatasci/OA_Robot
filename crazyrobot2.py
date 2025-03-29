import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


#region function forcecharts
def forcecharts(positions, times, path1, force1, velocity1, stoptodrop): #, mass1, h, beta):
    """    Function to plot the trajectory of the robots and the forces applied to them.
    Args:
        positions (list): List of positions for the robots.
        times (list): List of times for stopping and final positions.
        path1 (cvxpy.Variable): Optimized trajectory of Robot 1.
        force1 (cvxpy.Variable): Optimized forces applied to Robot 1.
    """
    # Create a figure with two subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=False)

#region Plot 1 - positions
#--------------------- Plot 1: Optimized trajectories of the Robots ----------------------------------------
    opt_R1 = path1.value


    axs[0].plot(opt_R1[:, 0], opt_R1[:, 1], 'b-', label="Trajectory of Robot 1", zorder=1)

    step = int(1 / h)  # Number of steps per second
    selected_indices = np.arange(0, len(opt_R1), step)  # Get indices for whole seconds

    # Plot only those points
    axs[0].plot(opt_R1[selected_indices, 0], opt_R1[selected_indices, 1], 'r*-', label="Trajectory of Robot 1", zorder=0)


    #axs[0].plot(opt_R1[:, 0], opt_R1[:, 1], 'bo-', label="Trajectory of Robot 1", zorder=1)
    axs[0].scatter([positions[0][0]], [positions[0][1]], c='green', marker='s', label="Initial Positions", zorder=3)
    axs[0].scatter([positions[-1][0]], [positions[-1][1]], c='red', marker='s', label="Final Positions", zorder=3)
    axs[0].scatter([pos[0] for pos in positions[1:-1]], [pos[1] for pos in positions[1:-1]], c='yellow', marker='o', label="Stopping Stations", zorder=2)

    axs[0].set_ylabel("Y Position")
    axs[0].legend()
    axs[0].set_title = "Robot Trajectory, optimized for minimum force"
    axs[0].grid()
    #endregion

#region Plot 2 - forces
# --------------------- Plot 2: optimal forces deployed to the Robots --------------------------

    # Time steps
    time_steps1 = np.arange(len(path1.value))

    # Compute force magnitudes
    opt_F1 = force1.value
    F1_magnitude = np.linalg.norm(opt_F1, axis=1)


    axs[1].plot(time_steps1*h, F1_magnitude, 'b-', label="Force Magnitude (Robot 1)", linewidth=2)
    axs[1].set_ylabel("Force Magnitude")
    axs[1].set_title("Force Magnitudes Over Time")
    if len(positions) > 2:
        stopstation=1
        for tstop in times[:-1]:
            axs[1].axvline(x=tstop*h, color='yellow', linestyle='--', label=f"Stopping Station #{stopstation}", zorder=2)
            stopstation += 1

    axs[1].legend()
    axs[1].grid()

    plt.suptitle(f"Trajectory and Force Magnitudes, h={h}, stop={stoptodrop}", fontsize=16)

  
    opt_F1 = force1.value

    # Time steps
    time_steps1 = np.arange(len(path1.value))
	#endregion

#region Plot 3 - velocity
# --------------------- Plot 3: Velocity of the robot --------------------------

    # Compute force magnitudes
    F1_magnitude = np.linalg.norm(opt_F1, axis=1)


    opt_Vel1 = cp.norm(velocity1.value, axis=1).value
    axs[2].plot(time_steps1*h, opt_Vel1, 'g-', label="Velocity (Robot 1)", linewidth=2)
    #axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Velocity")
    axs[2].set_title("Velocity Over Time")
    if len(positions) > 2:
        stopstation=1
        for tstop in times[:-1]:
            axs[2].axvline(x=tstop*h, color='yellow', linestyle='--', label=f"Stopping Station #{stopstation}", zorder=2)
            stopstation += 1

    axs[2].legend()
    axs[2].grid()
	#endregion

    plt.show()

#endregion function forcecharts


#region function print_results
def print_results(result, force1, mass1, h, beta, stoptodrop):
    opt_F1 = force1.value

    print(f"**** Sampling interval h={h}, inertia beta = {beta}")
    print(f"*** Optimal Forces for Robot 1: {opt_F1[:2]} ...")
    
    indivforces = [np.linalg.norm(forceind) for forceind in opt_F1]
    #indivforces = [np.linalg.norm(opt_F1[i]) for i in range(len(opt_F1))]
    print(f"*** Individual forces: {indivforces[:2]} ...")
#endregion function print_results


#region function optimize_robots
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
    
    #region Calculate variables for velocity and position at specific moments in time; adjusted for h
    # Initial, Stopping Station Positions, and Final Positions (Points in R^2)
    
    times = [int(x/h) for x in times]  # Convert time to number of steps
    
    initpos1 = np.array(positions[0])  # Initial position for the Robot
    if len(positions) > 2:
        stoppos1 = np.array(positions[1:-1])
    else:
        stoppos1 = np.nan
    finalpos1 = np.array(positions[-1])  # Final position for the Robot

    # Particular instant where Robot should rest at the corresponding position
    if len(times) > 1:
        stoptime1 = times[0:-1]  # Time at stopping station constraint for Robot
    else:
        stoptime1 = np.nan
    finaltime1 = times[-1]  # Final time--when the Robos should rest at the final station

    if len(positions) > 2:
        mass1 = [massrobot1 + masspackage1 if i <= stoptime1[0] else massrobot1 for i in range(finaltime1)]
    else:
        mass1 = [massrobot1+masspackage1 for i in range(finaltime1)]

#endregion

    #region --------------------------Optimization Variables----------------------
    path1 = cp.Variable((finaltime1, 2))  # Trajectory of Robot 1
    velocity1 = cp.Variable((finaltime1, 2))  # Velocity over time of Robot 1
    force1 = cp.Variable((finaltime1, 2))  # Force applied to Robot 1 over time
    #endregion

    #region ------------------------------Constraints---------------------------
    constraints = []

    # Initial conditions
    constraints += [path1[0] == initpos1]
    constraints += [velocity1[0] == 0]

    # Final conditions
    constraints += [path1[finaltime1-1] == finalpos1]
    constraints += [velocity1[finaltime1-1] == 0]

    # Stopping station constraints, if any
    if len(positions) > 2:
        # For each stopping station, set the path at the corresponding time to be the same as the stopping position
        for i in range(len(stoptime1)):
            constraints += [path1[stoptime1[i]] == stoppos1[i]]
            if stoptodrop:
                constraints += [velocity1[stoptime1[i]] == 0]  # Set velocity to zero at stopping station

    
    for t in range(finaltime1 - 1):
        constraints += [
            velocity1[t+1] == velocity1[t] + h * (force1[t] - beta * velocity1[t]) / mass1[t],
            path1[t+1] == path1[t] + h * velocity1[t]
        ]
    #endregion
  
    #region --------------------- objective function -----------------------------
    
    objective = cp.Minimize(
        cp.sum(cp.sum_squares(force1) / mass1)
        )
    #endregion
    
    #region --- Solve the optimization problem:
    # Minimize the objective function subject to the constraints
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS)
    #endregion

    #region Present the results
    print("Problem:")
    print(f"Solver Status: {prob.status}")

    # Check if solution was found before plotting
    if prob.status not in ["infeasible", "unbounded"]:
        print_results(result, force1, mass1, h, beta, stoptodrop)
        forcecharts(positions, times, path1, force1, velocity1, stoptodrop)

    else:
        print("Optimization problem is infeasible or unbounded.")
    #endregion

#endregion function optimize_robots


for h in [0.01]:
    optimize_robots(stoptodrop=False, h=h, positions=[[0,0], [7,7]], times=[8])
    optimize_robots(stoptodrop=False, h=h, positions=[[0,0], [3,3], [10,2],[9,8], [7,7]], times=[5, 20, 24, 30])
    optimize_robots(stoptodrop=True, h=h, positions=[[0,0], [3,3], [10,2],[9,8], [7,7]], times=[5, 20, 24, 30])
