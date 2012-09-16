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

#ifdef KSPACE_CLASS

KSpaceStyle(pppm/omp,PPPMOMP)

#else

#ifndef LMP_PPPM_OMP_H
#define LMP_PPPM_OMP_H

#include "pppm.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

  class PPPMOMP : public PPPM, public ThrOMP {
 public:
  PPPMOMP(class LAMMPS *, int, char **);
  virtual ~PPPMOMP () {};
  virtual void setup();
  virtual void compute(int, int);

 protected:
  virtual void allocate();
  virtual void deallocate();
  virtual void fieldforce();
  virtual void fieldforce_peratom();
  virtual void make_rho();

  void compute_rho1d_thr(FFT_SCALAR * const * const, const FFT_SCALAR &,
                         const FFT_SCALAR &, const FFT_SCALAR &);
//  void compute_rho_coeff();
//  void slabcorr(int);

};

}

#endif
#endif
