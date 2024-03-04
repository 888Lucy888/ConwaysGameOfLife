"""
conway.py 
A simple Python/matplotlib implementation of Conway's Game of Life.

@author: Gabriel Castillo - Base Code
@author: Jessica Fernanda Isunza López
@author: Celia Lucia Castañeda Arizaga
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from datetime import datetime

ON = 255
OFF = 0
vals = [ON, OFF]

CELL_TYPES = {
    "Block": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "Beehive": [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2)],
    "Loaf": [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 3), (3, 2)],
    "Boat": [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)],
    "Tub": [(0, 1), (1, 0), (1, 2), (2, 1)],
    "Blinker": [(0, 1), (1, 1), (2, 1)],
    "Toad": [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
    "Beacon": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)],
    "Glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "Lightweight Spaceship": [(0, 1), (0, 4), (1, 0), (2, 0), (3, 0), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4)]
}

def randomGrid(N):
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N, N)

def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0,    0, 255], 
                       [255,  0, 255], 
                       [0,  255, 255]])
    grid[i:i+3, j:j+3] = glider

def update(frameNum, img, grid, w, h):
    newGrid = grid.copy()
    for i in range(w):
        for j in range(h):
            nbs = 0
            for x in range(i - 1, i + 2):
                for y in range(j - 1, j + 2):
                    # Ensure toroidal 
                    # boundary conditions
                    nw, nh = x % w, y % h
                    if grid[nh, nw] == ON and (nw != i or nh != j):
                        nbs += 1

            # Apply Conway's Game of Life rules
            if grid[j, i] == ON:
                if nbs < 2 or nbs > 3:
                    newGrid[j, i] = OFF
            else:
                if nbs == 3:
                    newGrid[j, i] = ON

    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img,


#recieve/read file
def readFile(file_path):
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

def createGrid(width, height, living_cells):
    grid = np.zeros((height, width), dtype=int)
    for cell in living_cells:
        grid[cell[1]][cell[0]] = ON
    return grid

def countConfigs(grid, frame_num):
    configCount = {entity: 0 for entity in CELL_TYPES.keys()}
    
    for entity, coordinates in CELL_TYPES.items():
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                entity_found = True
                for dx, dy in coordinates:
                    if (x + dx < 0 or x + dx >= grid.shape[1]) or (y + dy < 0 or y + dy >= grid.shape[0]) or (grid[y + dy][x + dx] != ON):
                        entity_found = False
                        break
                if entity_found:
                    configCount[entity] += 1

    return configCount


# main() function
def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life system.py.")
    # TODO: add arguments
    filePath = input("File Path:")
    #read file
    file_path = filePath
    w, h, gens, living_cells = readFile(file_path)
        
    # set animation update interval
    updateInterval = 1000

    # declare grid
    grid = np.array([])
    # populate grid with random on/off - more off than on
    #grid = randomGrid(N)
    grid = createGrid(w, h, living_cells)
    # Uncomment lines to see the "glider" demo
    #grid = np.zeros(N*N).reshape(N, N)
    #addGlider(1, 1, grid)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, w, h),
                                  frames=gens,
                                  interval=updateInterval,
                                  save_count=50)
    plt.show()

    current_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Simulation at {current_date}")
    print(f"Universe size: {w} x {h}\n\n Results in simulation_results_{current_date}.txt")

    #file
    output_file = "simulation_results_"+current_date+".txt"
    with open(output_file, 'w') as file:
        file.write(f"Simulation at {current_date}")
        file.write(f"Universe size: {w} x {h}\n\n")

        # Count Configs
        for i in range(gens):
            entity_counts = countConfigs(grid, i)
            total_cells = sum(entity_counts.values())
            file.write(f"Iteration: {i + 1}\n")
            file.write("-" * 35 + "\n")
            file.write("| {:<20} | {:<10} | {:<10} |\n".format("Entity", "Count", "Percent"))
            file.write("-" * 35 + "\n")
            for entity, count in entity_counts.items():
                percent = (count / total_cells) * 100 if total_cells != 0 else 0
                file.write("| {:<20} | {:<10} | {:<10.2f}% |\n".format(entity, count, percent))
            file.write("-" * 35 + "\n\n")
            update(None, img, grid, w, h)



# call main
if __name__ == '__main__':
    main()