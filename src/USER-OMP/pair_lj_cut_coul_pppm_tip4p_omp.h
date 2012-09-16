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

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(lj/cut/coul/pppm/tip4p/omp,PairLJCutCoulPPPMTIP4POMP)

#else

#ifndef LMP_PAIR_LJ_CUT_COUL_PPPM_TIP4P_OMP_H
#define LMP_PAIR_LJ_CUT_COUL_PPPM_TIP4P_OMP_H

#include "pair_lj_cut_coul_long_tip4p.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class PairLJCutCoulPPPMTIP4POMP : public PairLJCutCoulLongTIP4P, public ThrOMP {

 public:
  PairLJCutCoulPPPMTIP4POMP(class LAMMPS *);
  virtual ~PairLJCutCoulPPPMTIP4POMP() {};

  virtual void init_style();

  virtual void compute(int, int);
  virtual double memory_usage();

 private:
  template <int, int, int, int>
  void eval(int ifrom, int ito, ThrData * const thr);
  void compute_newsite_thr(const double *, const double *,
                           const double *, double *) const;

  class PPPMTIP4PProxy *kspace;
  int nproxy;
};

}

#endif
#endif
