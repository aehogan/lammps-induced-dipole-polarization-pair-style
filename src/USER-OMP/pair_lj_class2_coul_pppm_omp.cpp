/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "math.h"
#include "pair_lj_class2_coul_pppm_omp.h"
#include "pppm_proxy.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"

#include <string.h>

#include "suffix.h"
using namespace LAMMPS_NS;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairLJClass2CoulPPPMOMP::PairLJClass2CoulPPPMOMP(LAMMPS *lmp) :
  PairLJClass2CoulLong(lmp), ThrOMP(lmp, THR_PAIR|THR_PROXY)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
  nproxy=1;

  kspace = NULL;
}

/* ---------------------------------------------------------------------- */

void PairLJClass2CoulPPPMOMP::init_style()
{
  if (comm->nthreads < 2)
    error->all(FLERR,"need at least two threads per MPI task for this pair style");

  if (strcmp(force->kspace_style,"pppm/proxy") != 0)
    error->all(FLERR,"kspace style pppm/proxy is required with this pair style");

  kspace = static_cast<PPPMProxy *>(force->kspace);

  PairLJClass2CoulLong::init_style();
}

/* ---------------------------------------------------------------------- */

void PairLJClass2CoulPPPMOMP::compute(int eflag, int vflag)
{
  if (eflag || vflag) {
    ev_setup(eflag,vflag);
  } else evflag = vflag_fdotr = 0;

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel default(none) shared(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads, nproxy);
    ThrData *thr = fix->get_thr(tid);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, thr);

    // thread id 0 runs pppm, the rest the pair style
    if (tid < nproxy) {
      kspace->compute_proxy(eflag,vflag);
    } else {
      if (evflag) {
        if (eflag) {
          if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
          else eval<1,1,0>(ifrom, ito, thr);
        } else {
          if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
          else eval<1,0,0>(ifrom, ito, thr);
        }
      } else {
        if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
        else eval<0,0,0>(ifrom, ito, thr);
      }
    }

    sync_threads();
    reduce_thr(this, eflag, vflag, thr, nproxy);
  } // end of omp parallel region
}

/* ---------------------------------------------------------------------- */

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairLJClass2CoulPPPMOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  int i,j,ii,jj,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double r,rsq,rinv,r2inv,r3inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = ecoul = 0.0;

  const double * const * const x = atom->x;
  double * const * const f = thr->get_f();
  const double * const q = atom->q;
  const int * const type = atom->type;
  const int nlocal = atom->nlocal;
  const double * const special_coul = force->special_coul;
  const double * const special_lj = force->special_lj;
  const double qqrd2e = force->qqrd2e;
  double fxtmp,fytmp,fztmp;

  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = iifrom; ii < iito; ++ii) {

    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    fxtmp=fytmp=fztmp=0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;

        if (rsq < cut_coulsq) {
          r = sqrt(rsq);
          grij = g_ewald * r;
          expm2 = exp(-grij*grij);
          t = 1.0 / (1.0 + EWALD_P*grij);
          erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
          prefactor = qqrd2e * qtmp*q[j]/r;
          forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
          if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else forcecoul = 0.0;

        if (rsq < cut_ljsq[itype][jtype]) {
          rinv = sqrt(r2inv);
          r3inv = r2inv*rinv;
          r6inv = r3inv*r3inv;
          forcelj = r6inv * (lj1[itype][jtype]*r3inv - lj2[itype][jtype]);
          forcelj *= factor_lj;
        } else forcelj = 0.0;

        fpair = (forcecoul + forcelj) * r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;
        if (NEWTON_PAIR || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (EFLAG) {
          if (rsq < cut_coulsq) {
            ecoul = prefactor*erfc;
            if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
          } else ecoul = 0.0;

          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv*(lj3[itype][jtype]*r3inv-lj4[itype][jtype]) -
              offset[itype][jtype];
            evdwl *= factor_lj;
          } else evdwl = 0.0;
        }

        if (EVFLAG) ev_tally_thr(this, i,j,nlocal,NEWTON_PAIR,
                                 evdwl,ecoul,fpair,delx,dely,delz,thr);
      }
    }
    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

double PairLJClass2CoulPPPMOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairLJClass2CoulLong::memory_usage();

  return bytes;
}
