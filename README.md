# AutoPhotonicsLab
FLAR - Code for the automated design of photonic surfaces in a self-driving lab

# Random Experiment Simulator

Overview

This repository contains two Python scripts that simulate an experiment by randomly selecting laser parameters and generating corresponding emissivity curves. The chosen parameters are then saved to a file for further analysis.

Prerequisites

- NumPy

File Descriptions

1. Sample Parameters Script:
   - This script loads laser parameters from a .npy file, randomly selects a batch of parameters, and saves them to a text file. The outputs are Power, Speed, Spacing.

    import numpy as np
    import os
    import sys
    ```python
    laser_parameters = np.load('inconel_data/input_test_data.npy')
    n = 10  # batch size
    idx = np.random.choice(np.arange(0, len(laser_parameters), 1), size=n, replace=False)  # random selection
    parameters_sample = laser_parameters[idx]  # sampled parameters
    directory = "sampled_data"
    file_path_par = os.path.join(directory, "sampled_parameters.txt")

    if not os.path.exists(directory):
        os.makedirs(directory)

    np.savetxt(file_path_par, parameters_sample)  # save the parameters in the sampled_data folder
    ```

2. Emissivity Curves Script:
   - This script simulates the emissivity curves in the same way that the parameters are simulated.

   Note: The code is similar to the Sample Parameters Script and will also include the logic to extract or generate required parameters.

Running the Scripts


1. Make sure you have the inconel_data folder and files in the appropriate directory.

2. Run the Sample Parameters Script:

   python sample_parameters.py

3. Run the Emissivity Curves Script:

   python emissivity_curves.py

Output

The selected parameters will be saved in the sampled_data directory as sampled_parameters.txt. You can use this file for further analysis or exploration of the experimental setup.

