WHaLE: Wind Heuristics and Lifecycle Estimator
==============================================

Overview
~~~~~~~~
Runs analyses for offshore wind projects by utilizing ORBIT (CapEx), WOMBAT (OpEx), and FLORIS (AEP)
to estimate the lifecycle costs using NREL's flagship technoeconomic models.


Requirements
~~~~~~~~~~~~
- Python 3.10


Environment Setup
~~~~~~~~~~~~~~~~~

Download the latest version of `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
for the appropriate OS. Follow the remaining `steps <https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
for the appropriate OS version.

Using conda, create a new virtual environment:


   `conda create -n <environment_name> python=3.10 --no-default-packages`
   
   `conda activate <environment_name>`
   
   `conda install -c anaconda pip`
   

   # to deactivate
   
   `conda deactivate`
   


General
~~~~~~~

This guide assumes that you have an environment that is already created and in use.

1. In the terminal/Anaconda Prompt, you should move to the directory where you want to put the project data/code:

    `cd my_path/`

2. Clone the project repository

    `git clone https://github.nrel.gov/OffshoreAnalysis/WHaLE.git`

3. Enter the directory

    `cd WHaLE`

4. Checkout the appropriate branch (most likely develop, unless working with a "release" version)

    `git checkout analysis/flowin_umaine`

5. Install WHaLE:

    `pip install -e .`

Custom installation for WOMBAT, ORBIT, and FLORIS updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is designed for working with in-progress updates to ORBIT and WOMBAT, and that the above steps have been completed. Please note that these repository version should not be updated unless the instructions are updated, nor should they be used for other work as this is to maintain a consistent workflow between users on this one project.

1. Uninstall ORBIT, WOMBAT and FLORIS

    `pip uninstall orbit-nrel wombat floris`

2. Clone ORBIT and WOMBAT to a separate working directory outside of the WHaLE repository folder like in step 2 above

   1. clone ORBIT
   
    `git clone https://github.com/WISDEM/ORBIT.git`
   
   2. clone WOMBAT
   
    `git clone https://github.com/WISDEM/WOMBAT.git`
   
   3. clone FLORIS
   
    `git clone https://github.com/NREL/floris.git`
   
3. Install the working version of ORBIT

   1. Enter the directory
   
    `cd ORBIT`

   2. Checkout the dev branch
   
    `git checkout dev`

   3. Install a local version that is able to be updated as the working branch changes or edits are made and downloaded
   
    `pip install -e .`
    
   4. Install last version of ORBIT
   
    `pip install orbit-nrel==1.0.8`

4. Install the working version of WOMBAT

   1. Enter the directory
   
    `cd WOMBAT`

   2. Checkout the develop branch
   
    `git checkout develop`

   3. Install a local version that is able to be updated as the working branch changes or edits are made and downloaded
   
    `pip install -e .`
      
5. Install the working version of FLORIS

   1. Enter the directory
   
    `cd floris`

   2. Checkout the main version of FLORIS
   
    `git checkout main`

   3. Install a local version that is able to be updated as the working branch changes or edits are made and downloaded
   
    `pip install -e .`
    
6. There is not enough  space in GH for weather profiles, make sure you copy and paste the appropiate weather profile in examples/library/weather
