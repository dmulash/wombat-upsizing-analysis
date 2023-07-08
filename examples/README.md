# Installation Guide for In-Progress Code Updates


## General

This guide assumes that you have an environment that is already created and in use.

1. In the terminal/Anaconda Prompt, you should move to the directory where you want to put the project data/code:

    `cd my_path/`

2. Clone the project repository

    `git clone https://github.nrel.gov/OffshoreAnalysis/WHaLE.git`

3. Enter the directory

    `cd WHaLE`

4. Checkout the appropriate branch (most likely develop, unless working with a "release" version)

    `git checkout develop`

5. Install WHaLE:

    `pip install -e .`

## Custom installation for WOMBAT and ORBIT updates

This is designed for working with in-progress updates to ORBIT and WOMBAT, and that the above steps have been completed. Please note that these repository version should not be updated unless the instructions are updated, nor should they be used for other work as this is to maintain a consistent workflow between users on this one project.

1. Uninstall ORBIT and WOMBAT

    `pip uninstall orbit-nrel wombat`

2. Clone ORBIT and WOMBAT to a separate working directory outside of the WHaLE repository folder like in step 2 above
   1. `git clone https://github.com/WISDEM/ORBIT.git`
   2. `git clonehttps://github.com/WISDEM/WOMBAT.git`

3. Install the working version of ORBIT
   1. Enter the directory

      `cd ORBIT`

   2. Checkout the electrical refactor branch

      `git checkout electrical-refactor`

   3. Install a local version that is able to be updated as the working branch changes or edits are made and downloaded

      `pip install -e .`

4. Install the working version of WOMBAT
   1. Enter the directory

      `cd WOMBAT`

   2. Checkout the second-to-last update to the develop branch (turns out the fix was only a partial fix--sorry!)

      `git checkout -f 45568d0`

   3. Install a local version that is able to be updated as the working branch changes or edits are made and downloaded

      `pip install -e .`
