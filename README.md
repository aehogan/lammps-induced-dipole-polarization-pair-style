This is the home of a pair style for LAMMPS that includes induced dipole polarization. The attribute "static_polarizability" is added with units of distance^3. The single pair_style lj/cut/coul/long/polarization command described below is also added. For more information and implementation details about induced dipole interactions see section 8 of the included pdf file "Theory and simulation of metal-organic materials and biomolecules" in the polarization folder.

**IMPORTANT: This pair style does not work with multiple processors.** Undefined behavior will happen if you attempt to use this pair style with more than one process.

## Syntax:

pair_style lj/cut/coul/long/polarization cutoff1 (cutoff2) keyword value ...

* *cutoff* = global cutoff for LJ (and Coulombic if only 1 arg) (distance units)

* *cutoff2* = global cutoff for Coulombic (optional) (distance units)

* zero or more keyword/value pairs may be appended

* *keyword* = precision or zodid or fixed_iteration or damp or max_iterations or damp_type or polar_gs or polar_gs_ranked or polar_gamma

* *precision* values = precision
precision = if fixed_iteration is disabled, keep iterating until the square of the change in all dipoles is less than precision

* *zodid* values = yes or no
yes/no = whether to only use the first approximation for the induced dipoles

* *fixed_iteration* values = yes or no
yes/no = whether to use fixed iteration or precision

* *damp* values = damp
damp = the damping parameter if using exponential dipole-dipole interaction damping

* *max_iterations* values = iterations
iterations = if using precision, the maximum number of iterations to be calculated before returning the first approximation, or, if using using fixed_iteration, the number of iterations to be calculated

* *damp_type* values = exponential or none
exponential = use exponential dipole-dipole interaction damping
none = don't use any dipole-dipole interaction damping

* *polar_gs* values = yes or no
yes/no = whether to use the Gauss-Seidel method to speed up convergence

* *polar_gs_ranked* values = yes or no
yes/no = whether to use the Gauss-Seidel method with a ranked array to speed up convergence

* *polar_gamma* values = gamma
gamma = number to precondition the dipoles with to speed up convergence

## Examples:

pair_style lj/cut/coul/long/polarization 2.5 12 max_iterations 30 damp_type exponential polar_gs_ranked yes

## Description:


**IMPORTANT:** This pair style does not work with multiple processors._ Undefined behavior will happen if you attempt to use this pair style with more than one process.

The pair style included in this repository adds explicit polarization using induced dipole interactions to the lj/cut/coul/long pair style.

The electric field created by static charges on other molecules as well as the induced electric field of every other dipole induces dipoles on every atom proportional to their static polarizabilty. The induced dipoles then interact with both the static charges and the other induced dipoles. Polarizabilities are set using the set command and have units of distance^3, for example,

set type 1 static_polarizability 2.19630

The static electric field is calculated using a shifted-force coulombic equation (similar to wolf with no damping).

For more information and implementation details about induced dipole interactions see section 8 of the included pdf file "Theory and simulation of metal-organic materials and biomolecules" in the polarization folder.

## Defaults:

* iterations_max = 50

* damping_type = none

* polar_damp = 2.1304

* zodid = no

* polar_precision = 0.00000000001

* fixed_iteration = no

* polar_gs = no

* polar_gs_ranked = yes

* polar_gamma = 1.03