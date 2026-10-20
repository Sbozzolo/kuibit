# CarpetX TSV fixtures

The data files are CarpetX or SpacetimeX output, not edited by hand.
Tests generate malformed and synthetic cases in temporary directories.

`boxinbox/` comes from a two-process, two-level CarpetX run.
The two parameter files are preserved in their output directories. Run the
initial parameter file and then the recovery parameter file with the same
executable and two MPI processes. The initial run covers iterations 0–4 and
checkpoints iteration 3; recovery continues from 3 to 6. The overlapping
iterations exercise restart handling. Level 1 has two separated boxes, so
the 1D x output includes a real gap.

The executable was built with CarpetX as of September 2026 (TSV output as in
revision `6c59144`) and AMReX 25.11. CoordinatesX `cell_coords` provides grid
values equal to their coordinates; BoxInBox supplies scalars and arrays. The
initial output has comment headers and separate iterations; recovery has plain
headers and combined iterations. The 3D files retain ghost points.

`testoutput/` contains CarpetX's `TestOutput/test/output-arrays` reference
output and parameter file at revision `6c59144`: one complex scalar and x/z
slices of a complex array.

`multipole/` contains the `mp_Psi4_l2_m2_r5.00.tsv` reference output from
SpacetimeX `Multipole/test/q1`.
