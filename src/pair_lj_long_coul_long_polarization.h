/* -*- c++ -*- ----------------------------------------------------------
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

PairStyle(lj/long/coul/long/polarization,PairLJLongCoulLongPolarization)

#else

#ifndef LMP_PAIR_LJ_LONG_COUL_LONG_POLARIZATION_H
#define LMP_PAIR_LJ_LONG_COUL_LONG_POLARIZATION_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJLongCoulLongPolarization : public Pair {
 public:
  double cut_coul;

  PairLJLongCoulLongPolarization(class LAMMPS *);
  virtual ~PairLJLongCoulLongPolarization();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);

  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

  void compute_inner();
  void compute_middle();
  void compute_outer(int, int);

 protected:
  double cut_lj_global;
  double **cut_lj, **cut_lj_read, **cut_ljsq;
  double cut_coulsq;
  double **epsilon_read, **epsilon, **sigma_read, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;
  double *cut_respa;
  double qdist;
  double g_ewald;
  double g_ewald_6;
  int ewald_order, ewald_off;

  void options(char **arg, int order);
  void allocate();

  /* polarization stuff */
  double **ef_induced;
  double **dipole_field_matrix;
  double **mu_induced_new,**mu_induced_old;
  double *rank_metric;
  double rmin;
  int *ranked_array;
  int nlocal_old;
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
  int use_previous;
  double polar_gamma;
  /* ------------------ */
};

}

#endif
#endif
