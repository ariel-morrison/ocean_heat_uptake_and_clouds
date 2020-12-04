# Repository for Southern Ocean heat uptake project
### Written by Dr. Ariel Morrison, Postdoctoral Research Fellow
### University of Victoria/School of Earth and Ocean Sciences, arielmorrison at uvic dot ca
### Collaboration with Dr. Hansi Singh (University of Victoria) and Dr. Phil Rasch (PNNL)

Supported by the University of Victoria and the U.S. Department of Energy.

*Description: The Southern Ocean has absorbed most of the excess heat associated with anthropogenic greenhouse gas emissions. Since the Southern Ocean is poorly observed, much of our knowledge of ocean heat uptake is based on climate model simulations. However, climate models do not agree on how atmospheric processes impact the mechanisms controlling Southern Ocean heat uptake. In some models the peak in Southern Ocean heat uptake coincides with a peak in absorbed shortwave radiation at the ocean surface, implying a decrease in cloud cover drives heat uptake. Other models indicate that Southern Ocean heat uptake is controlled by increasing energy transport into the region, possibly from more downwelling longwave radiation from increased cloud cover. Using reanalyses, satellite observations, and climate models, we ask: How do clouds and cloud properties affect Southern Ocean heat uptake over the past 40 years?*

*This code analyzes hourly data from the ECMWF (ERA-5) and Japanese Meteorological Agency (JRA-55) reanalyses from 1979-2019.*


**To download the code:**
`git clone https://github.com/ariel-morrison/ocean_heat_uptake_and_clouds`

**To RUN:**

**Do steps 1-2 the FIRST time you run the code:**

1) Install the virtual environment package to run code in a clean environment:

`python3 -m pip install --user virtualenv`

Note: If your pip package is out of date, update the pip package when prompted.


2) Create a clean virtual environment to download code requirements and run code. venv is the command to create the environment and env is the name of your environment:

`python3 -m venv env`

Note: If you get an error message "Error returned non-zero exit status 1," run this command without pip:

`python3 -m venv env --without-pip`



**If you've already installed a virtual environment for this project in your working directory, start here:**

3) Activate your virtual environment:

`source env/bin/activate`


4) Install requirements.txt (contains all required packages) using pip:

Example command:

`pip install -r requirements.txt`

Note: If a package needs to be uninstalled before the requirements can be installed, use `conda uninstall $package`


5) Run the script with user inputs:

`python southernOceanHeatUptakeClouds.py $working_dir $year $latitude $longitude $seaIce_cutoff`


Example commands:

- Running for the year 2016 and excluding all grid cells with an hourly mean sea ice concentration higher than 80%:
`python southernOceanHeatUptakeClouds.py "/global/cscratch1/sd/armorris" 2016 latitude longitude 0.8`


**User-defined inputs:**
1. Working directory, where all output will be saved and where data subdirectories are: $working_dir  --  e.g., "/global/cscratch1/sd/armorris" (put it in quotes)
2. Year
3. Latitude
4. Longitude
5. Sea ice concentration threshold for masking. All grid cells with an hourly mean sea ice concentration higher than the threshold value will be masked out of the analysis. To keep all grid cells, use 1 as threshold value: $seaIce_cutoff -- e.g., 0.75 (75% concentration)
