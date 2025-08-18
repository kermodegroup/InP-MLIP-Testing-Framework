# MLIP Tests
Test runners and plotting files for the MLIP tests:
- Lattice Constant
- Equation of State
- Elastic Constants
- Phonon Spectrum
- Point Defect Formation Energies
- Point Defect Migration Barriers
- Stacking Fault Energies
- Dislocation Quadrupole Formation Energies
- Dislocation Quadrupole Migration Barriers


# Key Files
active_model.py: Selects which model to run the test for, change the `active_model_name` variable to change model

DatasetErrs.py, ElasticConstants.py, EOS.py, Phonons.py, PointDefectFormation.py, PointDefectMigration.py, QuadrupoleFormation.py, QuadrupoleMigration.py, StackingFault.py: Test runners

*_plotting.py: Corresponding plotting scripts, which plot information from each model

DatasetTable.py: Generates a table of the configuration types in the dataset, table 1 in the publication

ComparisonTable.py: Generates a table of the MLIP comparisons, table 2 in the publication