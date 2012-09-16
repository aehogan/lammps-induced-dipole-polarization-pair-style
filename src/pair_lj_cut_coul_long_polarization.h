/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(lj/cut/coul/long/polarization,PairLJCutCoulLongPolarization)

#else

#ifndef LMP_PAIR_LJ_CUT_COUL_LONG_POLARIZATION_H
#define LMP_PAIR_LJ_CUT_COUL_LONG_POLARIZATION_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutCoulLongPolarization : public Pair {
 public:
  PairLJCutCoulLongPolarization(class LAMMPS *);
  virtual ~PairLJCutCoulLongPolarization();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);

  /*void compute_inner();
  void compute_middle();
  void compute_outer(int, int);*/
  void *extract(const char *, int &);

  /* polarization stuff */
  int pack_comm(int, int *, double *,int, int *);
  void unpack_comm(int, int, double *);
  /* ------------------ */

 protected:
  double cut_lj_global;
  double **cut_lj,**cut_ljsq;
  double cut_coul,cut_coulsq;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4,**offset;
  double *cut_respa;
  double g_ewald;

  double tabinnersq;
  double *rtable,*drtable,*ftable,*dftable,*ctable,*dctable;
  double *etable,*detable,*ptable,*dptable,*vtable,*dvtable;
  int ncoulshiftbits,ncoulmask;

  void allocate();
  void init_tables();
  void free_tables();

  /* polarization stuff */
  double **ef_induced;
  double **dipole_field_matrix;
  double **mu_induced_new,**mu_induced_old;
  double *dipole_rrms;
  double *rank_metric;
  double rmin;
  int *ranked_array;
  int ntotal_old,nlocal_old;
  int iterations_max;
  void build_dipole_field_matrix();
  int DipoleSolverIterative();
  int damping_type;
  int zodid;
  int debug;
  double polar_damp;
  double polar_precision;
  int fixed_iteration;
  int polar_gs,polar_gs_ranked;
  int polar_sor,polar_esor,polar_gamma;
  // minimum image stuff
  int polar_min_image;
  double *static_polarizability_total,*static_polarizability_local;
  double *rank_metric_total,*rank_metric_local;
  double *x_local_0,*x_local_1,*x_local_2,*x_total_0,*x_total_1,*x_total_2,**x_total;
  double *ef_local_0,*ef_local_1,*ef_local_2,*ef_total_0,*ef_total_1,*ef_total_2,**ef_total;
  double **mu_induced_total;
  int max_atoms_old,total_atoms_old;
  int *num_of_atoms,max_atoms,total_atoms;
  /* ------------------ */
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style lj/cut/coul/long/polarization requires atom attribute q

The atom style defined does not have this attribute.

E: Pair style is incompatible with KSpace style

If a pair style with a long-range Coulombic component is selected,
then a kspace style must also be used.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
