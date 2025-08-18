# InP MLIP Testing Framework
Dataset, models, and tests for benchmarking InP potentials

Tests:
- Lattice Constant
- Equation of State
- Elastic Constants
- Phonon Spectrum
- Point Defect Formation Energies
- Point Defect Migration Barriers
- Stacking Fault Energies
- Dislocation Quadrupole Formation Energies
- Dislocation Quadrupole Migration Barriers

## Installation
Basic requirements stored in `requirements.txt`. Also requires the LAMMPS Python module & executable installed with the `ML-PACE` (for ACE models), `ML-SNAP` (for EME-SNAP), and `MANYBODY` (for Vashishta) packages installed. Models are driven by the ASE `LAMMPSlib` calculator interface.


## Directory Structure
Dataset: Holds the InP Dataset in `.xyz` format, as well as the train and validation partitions used in MACE training

DFT_Convergence: Has details of the DFT convergence study to determine cutoff energy and k-point density

Potentials: Holds input files for each of the bespoke models tested: new ACE and MACE models trained on the dataset in `Dataset/`, as well as the Branicio Vashishta model and the EME-SNAP model

Test_Plots, Test_Results: Directories which hold the plots and results from the testing framework

Tests: Python files containing the tests for each potential