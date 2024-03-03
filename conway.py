"""
conway.py 
A simple Python/matplotlib implementation of Conway's Game of Life.
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

ON = 255
OFF = 0
vals = [ON, OFF]

def randomGrid(N):
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N, N)

def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0,    0, 255], 
                       [255,  0, 255], 
                       [0,  255, 255]])
    grid[i:i+3, j:j+3] = glider

def update(frameNum, img, grid, N):
    # copy grid since we require 8 neighbors for calculation
    # and we go line by line 
    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):
            # Compute 8-neghbor sum using toroidal boundary conditions - x and y wrap around
            # so that the simulation takes place on a toroidal surface.
            nbs = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                         grid[(i-1)%N, j] + grid[(i+1)%N, j] +
                         grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/255)
            
            # Rules
            if grid[i, j]  == ON: #alive or not (includes rule #2 automatically)
                # 1. if alive has less than 2 live neighbors dies, 
                # 3. if alive has more than 3 live neighbors dies
                if (nbs < 2) or (nbs > 3): 
                    newGrid[i, j] = OFF
            else:
                # 4. if dead has exactly 3 live neighbors revives
                if nbs == 3:
                    newGrid[i, j] = ON

    # update data
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img,

#recieve/read file
def input_file(file_path):
    living_cells = set()
    with open(file_path, 'r') as file:
        #dimensions
        dims_line = file.readline().strip().split()
        w, h = int(dims_line[0]), int(dims_line[1])  
            
        # generations
        gens_line = file.readline().strip().split()
        gens = int(gens_line[0])  
            
        # Read living cells positions
        for line in file:
            if line.strip():  
                cell_info = line.strip().split()
                x, y = int(cell_info[0]), int(cell_info[1]) 
                living_cells.add((x, y))
                    
    return w, h, gens, living_cells

# main() function
def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life system.py.")
    # TODO: add arguments
    
    # set grid size (number input)
    #N = 100
    universe_size = input("universe size: ")
    N = int(universe_size)

    #read file
    file_path = 'test.txt'
    w, h, gens, living_cells = input_file(file_path)
        
    # set animation update interval
    updateInterval = 50

    # declare grid
    grid = np.array([])
    # populate grid with random on/off - more off than on
    grid = randomGrid(N)
    # Uncomment lines to see the "glider" demo
    grid = np.zeros(N*N).reshape(N, N)
    addGlider(1, 1, grid)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, ),
                                  frames = 10,
                                  interval=updateInterval,
                                  save_count=50)

    plt.show()

# call main
if __name__ == '__main__':
    main()