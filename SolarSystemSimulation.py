
# coding: utf-8

# In[2]:

# Code Purpose: 
# This code simulates a rogue specified sized planet entering our solar system. The code simulates the orbits
# of all the planets starting on April 25th 2017. This code produces a 3D animation of the planet entering the
# solar system. This code also produces position and velocity arrays of all the planets for a specified length
# of time. 
#
# Notes: 
# There are many repeated lines of code for graphing planets with just slight changes. Loops could have been 
# used to reduce the amount of code, but they are not as time efficient.
#
#
# Inputs: 
# T = Specified amount of simulation time in years from April 25th, 2017
# Ts = Specified time step of simulation in days
# Rx = Rogue planets starting X-coordinate (Sun as Orgin) in Km
# Ry = Rogue planets starting Y-coordinate (Sun as Orgin) in Km
# Rz = Rogue planets starting Z-coordinate (Sun as Orgin) in Km
# Rvx = Rogue planets velocity vector in X direction in Km/Sec
# Rvy = Rogue planets velocity vector in Y direction in Km/Sec
# Rvz = Rogue planets velocity vector in Z direction in Km/Sec
# Rm = Rogue planets mass in Kg

#----------------------Defining some functions for use later in simulation-----------------------------------

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


def forceMagnitude(mi, mj, sep):
    """
    Compute magnitude of gravitational force between two particles.

    Parameters
    ----------
    mi, mj : float
        Particle masses in kg.
    sep : float
        Particle separation (distance between particles) in m.

    Returns
    -------
    force : float
        Gravitational force between particles in N.

    Example
    -------
        Input:
            mEarth = 6.0e24     # kg
            mPerson = 70.0      # kg
            radiusEarth = 6.4e6 # m
            print magnitudeOfForce(mEarth, mPerson, radiusEarth)
        Output:
            683.935546875
    """
    G = 6.67e-11                # m3 kg-1 s-2
    return G * mi * mj / sep**2 # N


def magnitude(vec):
    """
    Compute magnitude of any vector with an arbitrary number of elements.

    Parameters
    ----------
    vec : numpy array
        Any vector

    Returns
    -------
    magnitude : float
        The magnitude of that vector.

    Example
    -------
        Input:
            print magnitude(np.array([3.0, 4.0, 0.0]))
        Output:
            5.0
    """
    return np.sqrt(np.sum(vec**2))


def unitDirectionVector(pos_a, pos_b):
    """
    Create unit direction vector from pos_a to pos_b

    Parameters
    ----------
    pos_a, pos_b : two numpy arrays
        Any two vectors

    Returns
    -------
    unit direction vector : one numpy array (same size input vectors)
        The unit direction vector from pos_a toward pos_b

    Example
    -------
        Input:
            someplace = np.array([3.0,2.0,5.0])
            someplaceelse = np.array([1.0, -4.0, 8.0])
            print unitDirectionVector(someplace, someplaceelse)
        Output:
            [-0.28571429, -0.85714286,  0.42857143]
    """

    # calculate the separation between the two vectors
    separation = pos_b - pos_a

    # divide vector components by vector magnitude to make unit vector
    return separation/magnitude(separation)


def forceVector(mi, mj, pos_i, pos_j):
    """
    Compute gravitational force vector exerted on particle i by particle j.

    Parameters
    ----------
    mi, mj : floats
        Particle masses, in kg.
    pos_i, pos_j : numpy arrays
        Particle positions in cartesian coordinates, in m.

    Returns
    -------
    forceVec : numpy array
        Components of gravitational force vector, in N.

    Example
    -------
        Input:
            mEarth = 6.0e24     # kg
            mPerson = 70.0      # kg
            radiusEarth = 6.4e6 # m
            centerEarth = np.array([0,0,0])
            surfaceEarth = np.array([0,0,1])*radiusEarth
            print forceVector(mEarth, mPerson, centerEarth, surfaceEarth)

        Output:
            [   0.            0.          683.93554688]


    """

    # compute the magnitude of the distance between positions
    distance = magnitude(pos_i - pos_j)
    # this distance is in meters, because pos_i and pos_j were

    # compute the magnitude of the force
    force = forceMagnitude(mi, mj, distance)
    # the magnitude of the force is in Newtons

    # calculate the unit direction vector of the force
    direction = unitDirectionVector(pos_i, pos_j)
    # this vector is unitless, its magnitude should be 1.0

    return force*direction # a numpy array, with units of Newtons



# define a function to calculate force vectors for all particles
def calculateForceVectors(masses, positions):
    """
    Compute net gravitational force vectors on particles,
    given a list of masses and positions for all of them.

    Parameters
    ----------
    masses : list (or 1D numpy array) of floats
        Particle masses, in kg.
    positions : list (or numpy array) of 3-element numpy arrays
        Particle positions in cartesian coordinates, in meters,
        in the same order as the masses are listed. Each element
        in the list (a single particle's position) should be a
        3-element numpy array, referring to its X, Y, Z position.

    Returns
    -------
    forceVectrs : list of 3-element numpy arrays
        A list containing the net force vectors for each particles.
        Each element in the list is a 3-element numpy array that
        represents the net 3D force acting on a particle, after summing
        over the individual force vectors induced by every other particle.

    Example
    -------
        Input:
            au = 1.496e+11
            masses = [1.0e24, 40.0e24, 50.0e24, 30.0e24, 2.0e24]
            positions = [np.array([ 0.5,  2.6,  0.05])*au,
                         np.array([ 0.8,  9.1,  0.10])*au,
                         np.array([-4.1, -2.4,  0.80])*au,
                         np.array([10.7,  3.7,  0.00])*au,
                         np.array([-2.0, -1.9, -0.40])*au]

            # calculate and print the force vectors for all particles
            forces = calculateForceVectors(masses, positions)

            print '{:>10} | {:>10} | {:>10} | {:>10}'.format('particle', 'Fx', 'Fy', 'Fz')
            print '{:>10} | {:>10} | {:>10} | {:>10}'.format('(#)', '(N)', '(N)', '(N)')
            print '-'*49
            for i in range(len(forces)):
                Fx, Fy, Fz = forces[i]
                print '{:10.0f} | {:10.1e} | {:10.1e} | {:10.1e}'.format(i, Fx, Fy, Fz)

        Output:
              particle |         Fx |         Fy |         Fz
                   (#) |        (N) |        (N) |        (N)
            -------------------------------------------------
                     0 |   -1.3e+15 |    3.8e+14 |    3.5e+14
                     1 |    9.2e+15 |   -5.3e+16 |    1.8e+15
                     2 |    7.5e+16 |    5.4e+16 |   -2.7e+16
                     3 |   -4.2e+16 |    6.4e+15 |    1.1e+15
                     4 |   -4.0e+16 |   -7.5e+15 |    2.4e+16

        """

    # how many particles are there?
    N = len(positions)

    # create an empty list, which we will fill with force vectors
    forcevectors = []

    # loop over particles for which we want the force vector
    for i in range(N):

        # create a force vector with all three elements as zero
        vector = np.zeros(3)

        # loop over all the particles we need to include in the force sum
        for j in range(N):

            # as long as i and j are not the same...
            if j != i:

                # ...add in the force vector of particle j acting on particle i
                vector += forceVector(masses[i], masses[j], positions[i], positions[j])

        # append this force vector into the list of force vectors
        forcevectors.append(vector)

    # return the list of force vectors out of the function
    return forcevectors


def updateParticles(masses, positions, velocities, dt):
    """
    Evolve particles in time via leap-frog integrator scheme. This function
    takes masses, positions, velocities, and a time step dt as

    Parameters
    ----------
    masses : np.ndarray
        1-D array containing masses for all particles, in kg
        It has length N, where N is the number of particles.
    positions : np.ndarray
        2-D array containing (x, y, z) positions for all particles.
        Shape is (N, 3) where N is the number of particles.
    velocities : np.ndarray
        2-D array containing (x, y, z) velocities for all particles.
        Shape is (N, 3) where N is the number of particles.
    dt : float
        Evolve system for time dt (in seconds).

    Returns
    -------
    Updated particle positions and particle velocities, each being a 2-D
    array with shape (N, 3), where N is the number of particles.

    """

    startingPositions = np.array(positions)
    startingVelocities = np.array(velocities)

    # how many particles are there?
    nParticles, nDimensions = startingPositions.shape

    # make sure the three input arrays have consistent shapes
    assert(startingVelocities.shape == startingPositions.shape)
    assert(len(masses) == nParticles)

    # calculate net force vectors on all particles, at the starting position
    startingForces = np.array(calculateForceVectors(masses, startingPositions))

    # calculate the acceleration due to gravity, at the starting position
    startingAccelerations = startingForces/np.array(masses).reshape(nParticles, 1)

    # calculate the ending position
    nudge = startingVelocities*dt + 0.5*startingAccelerations*dt**2
    endingPositions = startingPositions + nudge

    # calculate net force vectors on all particles, at the ending position
    endingForces = np.array(calculateForceVectors(masses, endingPositions))

    # calculate the acceleration due to gravity, at the ending position
    endingAccelerations = endingForces/np.array(masses).reshape(nParticles, 1)

    # calculate the ending velocity
    endingVelocities = (startingVelocities +
                        0.5*(endingAccelerations + startingAccelerations)*dt)

    return endingPositions, endingVelocities


def calculateTrajectories(masses, initialPositions, initialVelocities, totalTime, timeStep):
    
    """
    Evolves particle initial positions and velocities in time. Starts with intial 
    positions and velocities, steps forward in time and calculates new velocities
    and positions. Stores data in an array
    
    
    Definitions:
    ----------------------------------------
    nParticles = Number of particles
    
    nDemensions = X, Y, Z coordinates of particles
    
    nTimes = Number of different time points to calculate positions and velocities
    
    
    Inputs:
    ----------------------------------------
    masses: 1D array with nParticle elements
    
    initialPositions: 2D array with nParticles and nDimensions elements
    
    initialVelocities: 2D array with nParticles and nDimensions elements
    
    totalTime: total time to evolve the system in seconds
    
    timeStep: size of each step of time in seconds.
    
    
    Outputs:
    ----------------------------------------
    Times : A 1D array containing the times used in seconds
    
    positionsAtTimes: A 3D array containing the number of particles, the x, y, z coordinates, and the different time points.
    
    velocitiesAtTimes: A 3D array containing the number of particles, the velocity vectors, and the different time points.
    
    """
    
    
    
    Times = np.arange(0, totalTime, timeStep)
    #Creates an array of times from 0 to the total time given with the specified increments 
    
    positionsAtTimes = np.array(initialPositions)
    velocitiesAtTimes = np.array(initialVelocities)
    #Create arrays to store data in

    
    
    for step, currentTime in enumerate(Times[:-1]):
        #Make a loop with step=the current step of the loop and currentTime= Current Time on the simulation
        
        if step == 0:
            
            currentPosition = initialPositions
            currentVelocity = initialVelocities
            #If the loop is running for the first time, use initial positions
            
        else:
        
            currentPosition = positionsAtTimes[:, :, step]
            currentVelocity = velocitiesAtTimes[:, :, step]
            #Create array of current positions and velocities for the timestep of the loop
        
        
        P, V = updateParticles(masses, currentPosition, currentVelocity, timeStep)
        #Calculate updated positions and velocities
        
        positionsAtTimes = np.dstack([positionsAtTimes, P])
        velocitiesAtTimes = np.dstack([velocitiesAtTimes, V])
        #add updated positions and velocities to arrays
        
    return Times, positionsAtTimes, velocitiesAtTimes
        
        
        


# In[3]:

# ----Now that the needed functions are defined, Time to write a function to simulate the data in 3D.---


T = 25.0
Ts = 3.0
#inputs for total time and time step

Rx = 1.429e9 * 1000.0 #meters
Ry = 1.429e9 * 1000.0 #meters
Rz = 0 * 1000.0 #meters
Rvx = -15 * 1000.0 #m/sec
Rvy = -5 * 1000.0 #m/sec
Rvz = 0 * 1000.0 #m/sec
Rm = 10e21 * 1.0 #kg
#input parameters for rogue planet

data = np.loadtxt('/home/hobbes/Documents/ASTR3800/SolarSystemAU')
#Load the initial conditions

masses, X_AU, Y_AU, Z_AU, VX_AU_Day, VY_AU_Day, VZ_AU_Day = data.T
#Transpose the different variable to different arrays

X = X_AU * 149600000000.0
Y = Y_AU * 149600000000.0
Z = Z_AU * 149600000000.0
#Convert all the positions from AU to meters

VX = VX_AU_Day * 149600000000 / 86400.0
VY = VY_AU_Day * 149600000000 / 86400.0
VZ = VZ_AU_Day * 149600000000 / 86400.0
#Convert from AU/Day to Meter/Sec

Position0 = np.array([X[0], Y[0], Z[0]])
Velocity0 = np.array([VX[0], VY[0], VZ[0]]) 
#Create initial position and velocity arrays for the Sun

Position1 = np.array([X[1], Y[1], Z[1]])
Velocity1 = np.array([VX[1], VY[1], VZ[1]]) 
#Create initial position and velocity arrays for Mercury

Position2 = np.array([X[2], Y[2], Z[2]])
Velocity2 = np.array([VX[2], VY[2], VZ[2]]) 
#Create initial position and velocity arrays for Venus

Position3 = np.array([X[3], Y[3], Z[3]])
Velocity3 = np.array([VX[3], VY[3], VZ[3]]) 
#Create initial position and velocity arrays for Earth

Position4 = np.array([X[4], Y[4], Z[4]])
Velocity4 = np.array([VX[4], VY[4], VZ[4]]) 
#Create initial position and velocity arrays for Mars

Position5 = np.array([X[5], Y[5], Z[5]])
Velocity5 = np.array([VX[5], VY[5], VZ[5]]) 
#Create initial position and velocity arrays for Jupiter

Position6 = np.array([X[6], Y[6], Z[6]])
Velocity6 = np.array([VX[6], VY[6], VZ[6]]) 
#Create initial position and velocity arrays for Saturn

Position7 = np.array([X[7], Y[7], Z[7]])
Velocity7 = np.array([VX[7], VY[7], VZ[7]]) 
#Create initial position and velocity arrays for Uranus

Position8 = np.array([X[8], Y[8], Z[8]])
Velocity8 = np.array([VX[8], VY[8], VZ[8]]) 
#Create initial position and velocity arrays for Neptune

Rogue_Position = np.array([Rx, Ry, Rz])
Rogue_Velocity = np.array([Rvx, Rvy, Rvz])
##Create initial position and velocity arrays for Rogue Planet

initialPositions = np.array([Position0, Position1, Position2, Position3, Position4, Position5, Position6, Position7, Position8, Rogue_Position])
initialVelocities = np.array([Velocity0, Velocity1, Velocity2, Velocity3, Velocity4, Velocity5, Velocity6, Velocity7, Velocity8, Rogue_Velocity])
#Create multidemensional initial arrays for position and velocities 

totalTime = 365.0 * T  * 24.0 * 60.0 * 60.0 # Specified Time Amount
#Convert Total Time to Seconds

timeStep = Ts * 24.0 * 60.0 * 60.0 # Specified time step
#Convert Time Step to Seconds 

masses = np.append(masses, Rm)
#Append Rm to mass array

#---------------------------Calculates trajectories using Nbody Code--------------
Times, Positions, Velocities = calculateTrajectories(masses, initialPositions, initialVelocities, totalTime, timeStep)
#CalculateTrajectories

Normalized_Mass = np.log(masses)
#Normalize my Masses for use in animation via log scale


print 'This simuation has {} steps.'.format(len(Times))


# In[ ]:

import matplotlib.animation as ani
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#Import some animation stuff


wri = ani.FFMpegWriter(bitrate=10000, fps=15)
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection = '3d')
#Set up parameters for animation


with wri.saving(fig, 'solarsystem.mp4', 100):
    
    angle = np.linspace(0, 0, len(Times))
    # You change the second variable to any roation you would like of the graph in degrees
    
    for step, currentTime in enumerate(Times):
        #Create a for loop to make an animation
        
        
        ax.cla()
        #Clear the plots
        
        
        Particle1_X= Positions[0, 0, step]
        Particle2_X= Positions[1, 0, step]
        Particle3_X= Positions[2, 0, step]
        Particle4_X= Positions[3, 0, step]
        Particle5_X= Positions[4, 0, step]
        Particle6_X= Positions[5, 0, step]
        Particle7_X= Positions[6, 0, step]
        Particle8_X= Positions[7, 0, step]
        Particle9_X= Positions[8, 0, step]
        Particle10_X= Positions[9, 0, step]
        #Create variables with X coordinates of planets


        Particle1_Y= Positions[0, 1, step]
        Particle2_Y= Positions[1, 1, step]
        Particle3_Y= Positions[2, 1, step]
        Particle4_Y= Positions[3, 1, step]
        Particle5_Y= Positions[4, 1, step]
        Particle6_Y= Positions[5, 1, step]
        Particle7_Y= Positions[6, 1, step]
        Particle8_Y= Positions[7, 1, step]
        Particle9_Y= Positions[8, 1, step]
        Particle10_Y= Positions[9, 1, step]
        #Create variables with Y coordinates of planets

        
        Particle1_Z= Positions[0, 2, step]
        Particle2_Z= Positions[1, 2, step]
        Particle3_Z= Positions[2, 2, step]
        Particle4_Z= Positions[3, 2, step]
        Particle5_Z= Positions[4, 2, step]
        Particle6_Z= Positions[5, 2, step]
        Particle7_Z= Positions[6, 2, step]
        Particle8_Z= Positions[7, 2, step]
        Particle9_Z= Positions[8, 2, step]
        Particle10_Z= Positions[9, 2, step]
        #Create variables with Z coordinates of planets 
        
        
        ax.set_xlim3d(np.min(Positions[:, 0, :]), np.max(Positions[:, 0, :]))
        ax.set_ylim3d(np.min(Positions[:, 1, :]), np.max(Positions[:, 1, :]))
        ax.set_zlim3d(np.min(Positions[:, 2, :]), np.max(Positions[:, 2, :]))
        #Set the limits of the axis for the plot

        
        ax.scatter(Particle1_X, Particle1_Y, Particle1_Z, label ='Sun', c='yellow', marker='o', s=Normalized_Mass[0])
        ax.scatter(Particle2_X, Particle2_Y, Particle2_Z, label ='Mercury', c='gray', marker='o', s=Normalized_Mass[1])
        ax.scatter(Particle3_X, Particle3_Y, Particle3_Z, label ='Venus', c='orange', marker='o', s=Normalized_Mass[2])
        ax.scatter(Particle4_X, Particle4_Y, Particle4_Z, label ='Earth', c='green', marker='o', s=Normalized_Mass[3])
        ax.scatter(Particle5_X, Particle5_Y, Particle5_Z, label ='Mars', c='red', marker='o', s=Normalized_Mass[4])
        ax.scatter(Particle6_X, Particle6_Y, Particle6_Z, label ='Jupiter', c='brown', marker='o', s=Normalized_Mass[5])
        ax.scatter(Particle7_X, Particle7_Y, Particle7_Z, label ='Saturn', c='gold', marker='o', s=Normalized_Mass[6])
        ax.scatter(Particle8_X, Particle8_Y, Particle8_Z, label ='Uranus', c='lightcyan', marker='o', s=Normalized_Mass[7])
        ax.scatter(Particle9_X, Particle9_Y, Particle9_Z, label ='Neptune', c='deepskyblue', marker='o', s=Normalized_Mass[8])
        ax.scatter(Particle10_X, Particle10_Y, Particle10_Z, label ='Rogue Planet', c='pink', marker='o', s=Normalized_Mass[9])
        #Plot the X, Y, Z coordinates of each of the objects (Y and Z are switched to make Y verticle and Z horizontal)
        
        
        plt.legend(loc=2)
        #Show the Legend

        
        ax.set_xlabel('X Coordinate of Masses in Meters')
        ax.set_ylabel('Y Coordinate of Masses in Meters')
        ax.set_zlabel('Z Coordinate of Masses in Meters')
        #Label the Axis

        
        ax.set_title('X-Y-Z Coordinates of Masses at {} Days Since April 25 2017'.format(currentTime/86400.0))
        #Create a title with current time in seconds
        
        
        ax.view_init(45, angle[step])
        plt.draw()
        

        
        if step % 10 == 0:
            #If we are on a step in the loop divisable by 20, then...
            
            print 'Currently at step {} out of {}'.format(step,len(Times))
            #For monitoring process
            
        if step % 2 == 0:
            #If we are on a step in the loop divisable by 2, then...

            wri.grab_frame()
            #Save the frame for the animation
        


# In[1]:




# In[ ]:



