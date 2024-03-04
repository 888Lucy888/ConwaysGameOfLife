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

ON = 1
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

CELL_TYPES_MASKS = {
    "Block": [[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 0]],

    "Beehive": [[0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0]],

    "Loaf": [[0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 0, 1, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0]],

    "Boat": [[0, 0, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],

    "Tub": [[0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]],

    "Blinker": [[[0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0]],

                [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]],

    "Toad": [[[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 0],
              [0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]],

    "Beacon": [[[0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0]],

               [[0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0]]],

    "Glider": [[[0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0]],

               [[0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]],

               [[0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0]],

               [[0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]]],

"Lightweight Spaceship": [[[0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]],

                           [[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]],

                           [[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0],
                            [0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]],

                            [[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 0, 1, 1, 0],
                            [0, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]]]
}

def count_mask_occurrences(grid, mask):
    rows = len(grid)
    cols = len(grid[0])
    mask_rows = len(mask)
    mask_cols = len(mask[0])
    count = 0

    for i in range(rows - mask_rows + 1):
        for j in range(cols - mask_cols + 1):
            # Extract subgrid
            subgrid = [row[j:j+mask_cols] for row in grid[i:i+mask_rows]]
            subgrid_list = [list(row) for row in subgrid]
            if subgrid_list == mask:
                count += 1

    return count

def is_mask_present(grid, mask):
    rows = len(grid)
    cols = len(grid[0])
    mask_rows = len(mask)
    mask_cols = len(mask[0])

    for i in range(rows - mask_rows + 1):
        for j in range(cols - mask_cols + 1):
            # Extract subgrid
            subgrid = [row[j:j+mask_cols] for row in grid[i:i+mask_rows]]
            # Check if subgrid matches the mask
            if np.array_equal(subgrid, mask):  # Use np.array_equal() for array comparison
                return True

    return False

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

    singles = ["Block", "Beehive", "Loaf", "Boat", "Tub"]

    #file
    output_file = "simulation_results_"+current_date+".txt"
    with open(output_file, 'w') as file:
        file.write(f"Simulation at {current_date}")
        file.write(f"Universe size: {w} x {h}\n\n")

        for i in range(gens):
            total = 0
            counted_figures = {}
            for figure, masks in CELL_TYPES_MASKS.items():
                occurrences = 0
                if figure in singles:
                    if is_mask_present(grid, masks):
                        print("Mask is present in the grid.")
                        occurrences += count_mask_occurrences(grid, masks)
                        print("Number of occurrences:", occurrences)
                else:
                    for mask in masks:
                        if is_mask_present(grid, mask):
                            print("Mask is present in the grid.")
                            occurrences += count_mask_occurrences(grid, mask)
                            print("Number of occurrences:", occurrences)
                counted_figures[figure] = occurrences
                total += occurrences
            file.write(f"Iteration: {i + 1}\n")
            file.write("-" * 35 + "\n")
            file.write("| {:<20} | {:<10} | {:<10} |\n".format("Entity", "Count", "Percent"))
            file.write("-" * 35 + "\n")
            for entity, count in counted_figures.items():
                percent = (count / total) * 100 if total != 0 else 0
                file.write("| {:<20} | {:<10} | {:<10.2f}% |\n".format(entity, count, percent))
            file.write("-" * 35 + "\n\n")
            update(None, img, grid, w, h)

        '''
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
            '''



# call main
if __name__ == '__main__':
    main()