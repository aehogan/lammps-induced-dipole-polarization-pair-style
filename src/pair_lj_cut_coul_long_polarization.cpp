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

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
                        Adam Hogan (USF)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_lj_cut_coul_long_polarization.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "kspace.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "float.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

enum{DAMPING_EXPONENTIAL,DAMPING_NONE};

/* ---------------------------------------------------------------------- */

PairLJCutCoulLongPolarization::PairLJCutCoulLongPolarization(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 1;
  respa_enable = 0;
  writedata = 1;
  ftable = NULL;
  qdist = 0.0;

  /* polarization stuff */
  /* set defaults */
  iterations_max = 50;
  damping_type = DAMPING_NONE;
  polar_damp = 2.1304;
  zodid = 0;
  polar_precision = 0.00000000001;
  fixed_iteration = 0;

  polar_gs = 0;
  polar_gs_ranked = 1;
  polar_gamma = 1.03;

  use_previous = 0;

  debug = 0;
  /* end defaults */

  /* create arrays */
  int nlocal = atom->nlocal;
  memory->create(ef_induced,nlocal,3,"pair:ef_induced");
  memory->create(mu_induced_new,nlocal,3,"pair:mu_induced_new");
  memory->create(mu_induced_old,nlocal,3,"pair:mu_induced_old");
  memory->create(dipole_field_matrix,3*nlocal,3*nlocal,"pair:dipole_field_matrix");
  memory->create(ranked_array,nlocal,"pair:ranked_array");
  memory->create(rank_metric,nlocal,"pair:rank_metric");
  nlocal_old = nlocal;
  /* end polarization stuff */
}

/* ---------------------------------------------------------------------- */

PairLJCutCoulLongPolarization::~PairLJCutCoulLongPolarization()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
  if (ftable) free_tables();

  /* polarization stuff */
  memory->destroy(ef_induced);
  memory->destroy(mu_induced_new);
  memory->destroy(mu_induced_old);
  memory->destroy(dipole_field_matrix);
  memory->destroy(ranked_array);
  memory->destroy(rank_metric);
  /* end polarization stuff */
}

/* ---------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::compute(int eflag, int vflag)
{
  /* polarization stuff */
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int i;

  /* reallocate arrays if number of atoms grew */
  if (nlocal > nlocal_old)
  {
    memory->destroy(ef_induced);
    memory->create(ef_induced,nlocal,3,"pair:ef_induced");
    memory->destroy(mu_induced_new);
    memory->create(mu_induced_new,nlocal,3,"pair:mu_induced_new");
    memory->destroy(mu_induced_old);
    memory->create(mu_induced_old,nlocal,3,"pair:mu_induced_old");
    memory->destroy(ranked_array);
    memory->create(ranked_array,nlocal,"pair:ranked_array");
    memory->destroy(dipole_field_matrix);
    memory->create(dipole_field_matrix,3*nlocal,3*nlocal,"pair:dipole_field_matrix");
    memory->destroy(rank_metric);
    memory->create(rank_metric,nlocal,"pair:rank_metric");
    nlocal_old = nlocal;
  }
  double **ef_static = atom->ef_static;
  for (i = 0; i < nlocal; i++)
  {
    ef_static[i][0] = 0;
    ef_static[i][1] = 0;
    ef_static[i][2] = 0;
  }
  double ef_temp;
  /* end polarization stuff */

  int /*i,*/ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  //int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double *static_polarizability = atom->static_polarizability;
  int *molecule = atom->molecule;

  /* polarization stuff */
  /* sort the dipoles most likey to change if using polar_gs_ranked */
  if (polar_gs_ranked) {
    /* communicate static polarizabilities */
    comm->forward_comm_pair(this);
    MPI_Barrier(world);
    rmin = 1000.0;
    for (i=0;i<nlocal;i++)
    {
      for (j=0;j<ntotal;j++)
      {
        if(i != j) {
          r = sqrt(pow(x[i][0]-x[j][0],2)+pow(x[i][1]-x[j][1],2)+pow(x[i][2]-x[j][2],2));
          if (static_polarizability[i]>0&&static_polarizability[j]>0&&rmin>r&&((molecule[i]!=molecule[j])||molecule[i]==0))
          {
            rmin = r;
          }
        }
      }
    }
    for (i=0;i<nlocal;i++)
    {
      rank_metric[i] = 0;
    }
    for (i=0;i<nlocal;i++)
    {
      for (j=0;j<ntotal;j++)
      {
        if(i != j) {
          r = sqrt(pow(x[i][0]-x[j][0],2)+pow(x[i][1]-x[j][1],2)+pow(x[i][2]-x[j][2],2));
          if (rmin*1.5>r&&((molecule[i]!=molecule[j])||molecule[i]==0))
          {
            rank_metric[i]+=static_polarizability[i]*static_polarizability[j];
          }
        }
      }
    }
  }
  /* end polarization stuff */

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

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
          if (!ncoultablebits || rsq <= tabinnersq) {
            r = sqrt(rsq);
            grij = g_ewald * r;
            expm2 = exp(-grij*grij);
            t = 1.0 / (1.0 + EWALD_P*grij);
            erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
            prefactor = qqrd2e * qtmp*q[j]/r;
            forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
          } else {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            itable = rsq_lookup.i & ncoulmask;
            itable >>= ncoulshiftbits;
            fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
            table = ftable[itable] + fraction*dftable[itable];
            forcecoul = qtmp*q[j] * table;
            if (factor_coul < 1.0) {
              table = ctable[itable] + fraction*dctable[itable];
              prefactor = qtmp*q[j] * table;
              forcecoul -= (1.0-factor_coul)*prefactor;
            }
          }
        } else forcecoul = 0.0;

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        } else forcelj = 0.0;

        fpair = (forcecoul + factor_lj*forcelj) * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          if (rsq < cut_coulsq) {
            if (!ncoultablebits || rsq <= tabinnersq)
              ecoul = prefactor*erfc;
            else {
              table = etable[itable] + fraction*detable[itable];
              ecoul = qtmp*q[j] * table;
            }
            if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
          } else ecoul = 0.0;

          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
              offset[itype][jtype];
            evdwl *= factor_lj;
          } else evdwl = 0.0;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  /* polarization stuff */
  double f_shift = -1.0/(cut_coul*cut_coul); 
  double dvdrr;
  double xjimage[3];

  /* calculate static electric field using minimum image */
  for (i = 0; i < nlocal; i++) {
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    for (j = i+1; j < nlocal; j++) {
      domain->closest_image(x[i],x[j],xjimage);
      delx = xtmp - xjimage[0];
      dely = ytmp - xjimage[1];
      delz = ztmp - xjimage[2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq <= cut_coulsq)
      {
        if ( (molecule[i]!=molecule[j])||molecule[i]==0 )
        {
          r = sqrt(rsq);

          /* Use wolf to calculate the electric field (no damping) */
          dvdrr = 1.0/rsq + f_shift;
          ef_temp = dvdrr*1.0/r;

          ef_static[i][0] += ef_temp*q[j]*delx;
          ef_static[i][1] += ef_temp*q[j]*dely;
          ef_static[i][2] += ef_temp*q[j]*delz;
          ef_static[j][0] -= ef_temp*qtmp*delx;
          ef_static[j][1] -= ef_temp*qtmp*dely;
          ef_static[j][2] -= ef_temp*qtmp*delz;
        }
      }
    }
  }

  double **mu_induced = atom->mu_induced;

  int p,iterations;

  double elementary_charge_to_sqrt_energy_length = sqrt(qqrd2e);

  /* set the static electric field and first guess to alpha*E */
  for (i = 0; i < nlocal; i++) {
    /* it is more convenient to work in gaussian-like units for charges and electric fields */
    ef_static[i][0] = ef_static[i][0]*elementary_charge_to_sqrt_energy_length;
    ef_static[i][1] = ef_static[i][1]*elementary_charge_to_sqrt_energy_length;
    ef_static[i][2] = ef_static[i][2]*elementary_charge_to_sqrt_energy_length;
    /* don't reset the induced dipoles if use_previous is on */
    if (!use_previous)
    {
      /* otherwise set it to alpha*E */
      mu_induced[i][0] = static_polarizability[i]*ef_static[i][0];
      mu_induced[i][1] = static_polarizability[i]*ef_static[i][1];
      mu_induced[i][2] = static_polarizability[i]*ef_static[i][2];
      mu_induced[i][0] *= polar_gamma;
      mu_induced[i][1] *= polar_gamma;
      mu_induced[i][2] *= polar_gamma;
    }
  }

  /* solve for the induced dipoles */
  if (!zodid) iterations = DipoleSolverIterative();
  else iterations = 0;
  if (debug) fprintf(screen,"iterations: %d\n",iterations);

  /* debugging energy calculation - not actually used, should be the same as the polarization energy
     calculated from the pairwise forces */
  double u_polar = 0.0;
  if (debug)
  {
    for (i=0;i<nlocal;i++)
    {
      u_polar += ef_static[i][0]*mu_induced[i][0] + ef_static[i][1]*mu_induced[i][1] + ef_static[i][2]*mu_induced[i][2];
    }
    u_polar *= -0.5;
    printf("u_polar: %.18f\n",u_polar);
  }

  /* variables for dipole forces */
  double forcecoulx,forcecouly,forcecoulz,fx,fy,fz;
  double r3inv,r5inv,r7inv,pdotp,pidotr,pjdotr,pre1,pre2,pre3,pre4,pre5;
  double ef_0,ef_1,ef_2;
  double **mu = mu_induced;
  double xsq,ysq,zsq,common_factor;
  double forcetotalx,forcetotaly,forcetotalz;
  double forcedipolex,forcedipoley,forcedipolez;
  double forceefx,forceefy,forceefz;
  forcetotalx = forcetotaly = forcetotalz = 0.0;
  forcedipolex = forcedipoley = forcedipolez = 0.0;
  forceefx = forceefy = forceefz = 0.0;

  /* dipole forces */
  u_polar = 0.0;
  double u_polar_self = 0.0;
  double u_polar_ef = 0.0;
  double u_polar_dd = 0.0;
  double term_1,term_2,term_3;
  for (i = 0; i < nlocal; i++) {
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    /* self interaction energy */
    if (eflag&&static_polarizability[i]!=0.0)
      u_polar_self += 0.5 * (mu[i][0]*mu[i][0]+mu[i][1]*mu[i][1]+mu[i][2]*mu[i][2])/static_polarizability[i];

    for (j = i+1; j < nlocal; j++) {
      /* using minimum image again to be consistent */
      domain->closest_image(x[i],x[j],xjimage);
      delx = xtmp - xjimage[0];
      dely = ytmp - xjimage[1];
      delz = ztmp - xjimage[2];
      xsq = delx*delx;
      ysq = dely*dely;
      zsq = delz*delz;
      rsq = xsq + ysq + zsq;

      r2inv = 1.0/rsq;

      double rinv = sqrt(r2inv);
      r = 1.0/rinv;
      r3inv = r2inv*rinv;

      forcecoulx = forcecouly = forcecoulz = 0.0;

      if (rsq < cut_coulsq)
      {

        if ( (molecule[i]!=molecule[j])||molecule[i]==0 )
        {
          /* using wolf again */
          dvdrr = 1.0/rsq + f_shift;
          ef_temp = dvdrr*1.0/r*elementary_charge_to_sqrt_energy_length;

          /* dipole on i, charge on j interaction */
          if (static_polarizability[i]!=0.0&&q[j]!=0.0)
          {
            common_factor = q[j]*elementary_charge_to_sqrt_energy_length*r3inv;
            forcecoulx += common_factor * (mu[i][0] * ((-2.0*xsq+ysq+zsq)*r2inv + f_shift*(ysq+zsq)) + \
                                           mu[i][1] * (-3.0*delx*dely*r2inv - f_shift*delx*dely) + \
                                           mu[i][2] * (-3.0*delx*delz*r2inv - f_shift*delx*delz));
            forcecouly += common_factor * (mu[i][0] * (-3.0*delx*dely*r2inv - f_shift*delx*dely) + \
                                           mu[i][1] * ((-2.0*ysq+xsq+zsq)*r2inv + f_shift*(xsq+zsq)) + \
                                           mu[i][2] * (-3.0*dely*delz*r2inv - f_shift*dely*delz));
            forcecoulz += common_factor * (mu[i][0] * (-3.0*delx*delz*r2inv - f_shift*delx*delz) + \
                                           mu[i][1] * (-3.0*dely*delz*r2inv - f_shift*dely*delz) + \
                                           mu[i][2] * ((-2.0*zsq+xsq+ysq)*r2inv + f_shift*(xsq+ysq)));
            if (eflag)
            {
              ef_0 = ef_temp*q[j]*delx;
              ef_1 = ef_temp*q[j]*dely;
              ef_2 = ef_temp*q[j]*delz;

              u_polar_ef -= mu[i][0]*ef_0 + mu[i][1]*ef_1 + mu[i][2]*ef_2;
            }
          }

          /* dipole on j, charge on i interaction */
          if (static_polarizability[j]!=0.0&&qtmp!=0.0)
          {
            common_factor = qtmp*elementary_charge_to_sqrt_energy_length*r3inv;
            forcecoulx -= common_factor * (mu[j][0] * ((-2.0*xsq+ysq+zsq)*r2inv + f_shift*(ysq+zsq)) + \
                                             mu[j][1] * (-3.0*delx*dely*r2inv - f_shift*delx*dely) + \
                                             mu[j][2] * (-3.0*delx*delz*r2inv - f_shift*delx*delz));
            forcecouly -= common_factor * (mu[j][0] * (-3.0*delx*dely*r2inv - f_shift*delx*dely) + \
                                             mu[j][1] * ((-2.0*ysq+xsq+zsq)*r2inv + f_shift*(xsq+zsq)) + \
                                             mu[j][2] * (-3.0*dely*delz*r2inv - f_shift*dely*delz));
            forcecoulz -= common_factor * (mu[j][0] * (-3.0*delx*delz*r2inv - f_shift*delx*delz) + \
                                             mu[j][1] * (-3.0*dely*delz*r2inv - f_shift*dely*delz) + \
                                             mu[j][2] * ((-2.0*zsq+xsq+ysq)*r2inv + f_shift*(xsq+ysq)));
            if (eflag)
            {
              ef_0 = ef_temp*qtmp*delx;
              ef_1 = ef_temp*qtmp*dely;
              ef_2 = ef_temp*qtmp*delz;

              u_polar_ef += mu[j][0]*ef_0 + mu[j][1]*ef_1 + mu[j][2]*ef_2;
            }
          }
        }
      }

      /* dipole on i, dipole on j interaction */
      if (static_polarizability[i]!=0.0 && static_polarizability[j]!=0.0)
      {
        /* exponential dipole-dipole damping */
        if(damping_type == DAMPING_EXPONENTIAL)
        {
          r5inv = r3inv*r2inv;
          r7inv = r5inv*r2inv;

          term_1 = exp(-polar_damp*r);
          term_2 = 1.0+polar_damp*r+0.5*polar_damp*polar_damp*r*r;
          term_3 = 1.0+polar_damp*r+0.5*polar_damp*polar_damp*r*r+1.0/6.0*polar_damp*polar_damp*polar_damp*r*r*r;

          pdotp = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];
          pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
          pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;

          pre1 = 3.0*r5inv*pdotp*(1.0-term_1*term_2) - 15.0*r7inv*pidotr*pjdotr*(1.0-term_1*term_3);
          pre2 = 3.0*r5inv*pjdotr*(1.0-term_1*term_3);
          pre3 = 3.0*r5inv*pidotr*(1.0-term_1*term_3);
          pre4 = -pdotp*r3inv*(-term_1*(polar_damp*rinv+polar_damp*polar_damp) + term_1*polar_damp*term_2*rinv);
          pre5 = 3.0*pidotr*pjdotr*r5inv*(-term_1*(polar_damp*rinv+polar_damp*polar_damp+0.5*r*polar_damp*polar_damp*polar_damp)+term_1*polar_damp*term_3*rinv);

          forcecoulx += pre1*delx + pre2*mu[i][0] + pre3*mu[j][0] + pre4*delx + pre5*delx;
          forcecouly += pre1*dely + pre2*mu[i][1] + pre3*mu[j][1] + pre4*dely + pre5*dely;
          forcecoulz += pre1*delz + pre2*mu[i][2] + pre3*mu[j][2] + pre4*delz + pre5*delz;

          if (eflag)
          {
            u_polar_dd += r3inv*pdotp*(1.0-term_1*term_2) - 3.0*r5inv*pidotr*pjdotr*(1.0-term_1*term_3);
          }

          /* debug information */
          if (debug)
          {
            if (i==0)
            {
              forcedipolex += pre1*delx + pre2*mu[i][0] + pre3*mu[j][0] + pre4*delx + pre5*delx;
              forcedipoley += pre1*dely + pre2*mu[i][1] + pre3*mu[j][1] + pre4*dely + pre5*dely;
              forcedipolez += pre1*delz + pre2*mu[i][2] + pre3*mu[j][2] + pre4*delz + pre5*delz;
            }
            if (j==0)
            {
              forcedipolex -= pre1*delx + pre2*mu[i][0] + pre3*mu[j][0] + pre4*delx + pre5*delx;
              forcedipoley -= pre1*dely + pre2*mu[i][1] + pre3*mu[j][1] + pre4*dely + pre5*dely;
              forcedipolez -= pre1*delz + pre2*mu[i][2] + pre3*mu[j][2] + pre4*delz + pre5*delz;
            }
          }
          /* ---------------- */
        }
        /* no dipole-dipole damping */
        else
        {
          r5inv = r3inv*r2inv;
          r7inv = r5inv*r2inv;

          pdotp = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];
          pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
          pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;

          pre1 = 3.0*r5inv*pdotp - 15.0*r7inv*pidotr*pjdotr;
          pre2 = 3.0*r5inv*pjdotr;
          pre3 = 3.0*r5inv*pidotr;

          forcecoulx += pre1*delx + pre2*mu[i][0] + pre3*mu[j][0];
          forcecouly += pre1*dely + pre2*mu[i][1] + pre3*mu[j][1];
          forcecoulz += pre1*delz + pre2*mu[i][2] + pre3*mu[j][2];

          if (eflag)
          {
            u_polar_dd += r3inv*pdotp - 3.0*r5inv*pidotr*pjdotr;
          }

          /* debug information */
          if (debug)
          {
            if (i==0)
            {
              forcedipolex += pre1*delx + pre2*mu[i][0] + pre3*mu[j][0];
              forcedipoley += pre1*dely + pre2*mu[i][1] + pre3*mu[j][1];
              forcedipolez += pre1*delz + pre2*mu[i][2] + pre3*mu[j][2];
            }
            if (j==0)
            {
              forcedipolex -= pre1*delx + pre2*mu[i][0] + pre3*mu[j][0];
              forcedipoley -= pre1*dely + pre2*mu[i][1] + pre3*mu[j][1];
              forcedipolez -= pre1*delz + pre2*mu[i][2] + pre3*mu[j][2];
            }
          }
          /* ---------------- */
        }
      }

      f[i][0] += forcecoulx;
      f[i][1] += forcecouly;
      f[i][2] += forcecoulz;

      if (newton_pair || j < nlocal)
      {
        f[j][0] -= forcecoulx;
        f[j][1] -= forcecouly;
        f[j][2] -= forcecoulz;
      }

      /* debug information */
      if (i==0)
      {
        forcetotalx += forcecoulx;
        forcetotaly += forcecouly;
        forcetotalz += forcecoulz;
      }
      if (j==0)
      {
        forcetotalx -= forcecoulx;
        forcetotaly -= forcecouly;
        forcetotalz -= forcecoulz;
      }
      /* ---------------- */
      if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,0.0,0.0,forcecoulx,forcecouly,forcecoulz,delx,dely,delz);
    }
  }
  u_polar = u_polar_self + u_polar_ef + u_polar_dd;
  if (debug)
  {
    printf("self: %.18f\nef: %.18f\ndd: %.18f\n",u_polar_self,u_polar_ef,u_polar_dd);
    printf("u_polar calc: %.18f\n",u_polar);
    printf("polar force on atom 0: %.18f,%.18f,%.18f\n",forcetotalx,forcetotaly,forcetotalz);
    printf("polar dipole force on atom 0: %.18f,%.18f,%.18f\n",forcedipolex,forcedipoley,forcedipolez);
    printf("pos of atom 0: %.5f,%.5f,%.5f\n",x[0][0],x[0][1],x[0][2]);
  }
  force->pair->eng_pol = u_polar;
  /* end polarization stuff */

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::settings(int narg, char **arg)
{
 if (narg < 1 ) error->all(FLERR,"Illegal pair_style command");

  cut_lj_global = force->numeric(FLERR,arg[0]);
  if (narg == 1) cut_coul = cut_lj_global;
  else cut_coul = force->numeric(FLERR,arg[1]);

  /* polarization stuff */
  int iarg;
  iarg = 2;
  while (iarg < narg)
  {
    if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style command");
    if (strcmp("precision",arg[iarg])==0)
    {
      polar_precision = force->numeric(FLERR,arg[iarg+1]);
    }
    else if (strcmp("zodid",arg[iarg])==0)
    {
      if (polar_gs||polar_gs_ranked) error->all(FLERR,"Zodid doesn't work with polar_gs or polar_gs_ranked");
      if (strcmp("yes",arg[iarg+1])==0) zodid = 1;
      else if (strcmp("no",arg[iarg+1])==0) zodid = 0;
      else error->all(FLERR,"Illegal pair_style command");
    }
    else if (strcmp("fixed_iteration",arg[iarg])==0)
    {
      if (strcmp("yes",arg[iarg+1])==0) fixed_iteration = 1;
      else if (strcmp("no",arg[iarg+1])==0) fixed_iteration = 0;
      else error->all(FLERR,"Illegal pair_style command");
    }
    else if (strcmp("damp",arg[iarg])==0)
    {
      polar_damp = force->numeric(FLERR,arg[iarg+1]);
    }
    else if (strcmp("max_iterations",arg[iarg])==0)
    {
      iterations_max = force->inumeric(FLERR,arg[iarg+1]);
    }
    else if (strcmp("damp_type",arg[iarg])==0)
    {
      if (strcmp("exponential",arg[iarg+1])==0) damping_type = DAMPING_EXPONENTIAL;
      else if (strcmp("none",arg[iarg+1])==0) damping_type = DAMPING_NONE;
      else error->all(FLERR,"Illegal pair_style command");
    }
    else if (strcmp("polar_gs",arg[iarg])==0)
    {
      if (polar_gs_ranked) error->all(FLERR,"polar_gs and polar_gs_ranked are mutually exclusive");
      if (strcmp("yes",arg[iarg+1])==0) polar_gs = 1;
      else if (strcmp("no",arg[iarg+1])==0) polar_gs = 0;
      else error->all(FLERR,"Illegal pair_style command");
    }
    else if (strcmp("polar_gs_ranked",arg[iarg])==0)
    {
      if (polar_gs) error->all(FLERR,"polar_gs and polar_gs_ranked are mutually exclusive");
      if (strcmp("yes",arg[iarg+1])==0) polar_gs_ranked = 1;
      else if (strcmp("no",arg[iarg+1])==0) polar_gs_ranked = 0;
      else error->all(FLERR,"Illegal pair_style command");
    }
    else if (strcmp("polar_gamma",arg[iarg])==0)
    {
      polar_gamma = force->numeric(FLERR,arg[iarg+1]);
    }
    else if (strcmp("debug",arg[iarg])==0)
    {
      if (strcmp("yes",arg[iarg+1])==0) debug = 1;
      else if (strcmp("no",arg[iarg+1])==0) debug = 0;
      else error->all(FLERR,"Illegal pair_style command");
    }
    else if (strcmp("use_previous",arg[iarg])==0)
    {
      if (strcmp("yes",arg[iarg+1])==0) use_previous = 1;
      else if (strcmp("no",arg[iarg+1])==0) use_previous = 0;
      else error->all(FLERR,"Illegal pair_style command");
    }
    else error->all(FLERR,"Illegal pair_style command");
    iarg+=2;
  }
  /* end polarization stuff */

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_lj_one = cut_lj_global;
  if (narg == 5) cut_lj_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut_lj[i][j] = cut_lj_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair style lj/cut/coul/long requires atom attribute q");

  /* polarization stuff */
  if (!atom->static_polarizability_flag)
    error->all(FLERR,"Pair style lj/cut/coul/long/polarization requires atom attribute polarizability");
  /* end polarization stuff */

  // request regular or rRESPA neighbor list

  int irequest;
  int respa = 0;

  if (update->whichflag == 1 && strstr(update->integrate_style,"respa")) {
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
  }

  irequest = neighbor->request(this,instance_me);

  if (respa >= 1) {
    neighbor->requests[irequest]->respaouter = 1;
    neighbor->requests[irequest]->respainner = 1;
  }
  if (respa == 2) neighbor->requests[irequest]->respamiddle = 1;

  cut_coulsq = cut_coul * cut_coul;

  // set rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0)
    cut_respa = ((Respa *) update->integrate)->cutoff;
  else cut_respa = NULL;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->all(FLERR,"Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;

  // setup force tables

  if (ncoultablebits) init_tables(cut_coul,cut_respa);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCutCoulLongPolarization::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i],cut_lj[j][j]);
  }

  // include TIP4P qdist in full cutoff, qdist = 0.0 if not TIP4P

  double cut = MAX(cut_lj[i][j],cut_coul+2.0*qdist);
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag && (cut_lj[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  cut_ljsq[j][i] = cut_ljsq[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && MIN(cut_lj[i][j],cut_coul) < cut_respa[3])
    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double sig2 = sigma[i][j]*sigma[i][j];
    double sig6 = sig2*sig2*sig2;
    double rc3 = cut_lj[i][j]*cut_lj[i][j]*cut_lj[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
  }

  return cut;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut_lj[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
  fwrite(&ncoultablebits,sizeof(int),1,fp);
  fwrite(&tabinner,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_lj_global,sizeof(double),1,fp);
    fread(&cut_coul,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
    fread(&ncoultablebits,sizeof(int),1,fp);
    fread(&tabinner,sizeof(double),1,fp);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
  MPI_Bcast(&ncoultablebits,1,MPI_INT,0,world);
  MPI_Bcast(&tabinner,1,MPI_DOUBLE,0,world);
}


/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],cut_lj[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLJCutCoulLongPolarization::single(int i, int j, int itype, int jtype,
                                 double rsq,
                                 double factor_coul, double factor_lj,
                                 double &fforce)
{
  double r2inv,r6inv,r,grij,expm2,t,erfc,prefactor;
  double fraction,table,forcecoul,forcelj,phicoul,philj;
  int itable;

  r2inv = 1.0/rsq;
  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq) {
      r = sqrt(rsq);
      grij = g_ewald * r;
      expm2 = exp(-grij*grij);
      t = 1.0 / (1.0 + EWALD_P*grij);
      erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
      prefactor = force->qqrd2e * atom->q[i]*atom->q[j]/r;
      forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
      if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
    } else {
      union_int_float_t rsq_lookup_single;
      rsq_lookup_single.f = rsq;
      itable = rsq_lookup_single.i & ncoulmask;
      itable >>= ncoulshiftbits;
      fraction = (rsq_lookup_single.f - rtable[itable]) * drtable[itable];
      table = ftable[itable] + fraction*dftable[itable];
      forcecoul = atom->q[i]*atom->q[j] * table;
      if (factor_coul < 1.0) {
        table = ctable[itable] + fraction*dctable[itable];
        prefactor = atom->q[i]*atom->q[j] * table;
        forcecoul -= (1.0-factor_coul)*prefactor;
      }
    }
  } else forcecoul = 0.0;

  if (rsq < cut_ljsq[itype][jtype]) {
    r6inv = r2inv*r2inv*r2inv;
    forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
  } else forcelj = 0.0;

  fforce = (forcecoul + factor_lj*forcelj) * r2inv;

  double eng = 0.0;
  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq)
      phicoul = prefactor*erfc;
    else {
      table = etable[itable] + fraction*detable[itable];
      phicoul = atom->q[i]*atom->q[j] * table;
    }
    if (factor_coul < 1.0) phicoul -= (1.0-factor_coul)*prefactor;
    eng += phicoul;
  }

  if (rsq < cut_ljsq[itype][jtype]) {
    philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
      offset[itype][jtype];
    eng += factor_lj*philj;
  }

  return eng;
}

/* ---------------------------------------------------------------------- */

void *PairLJCutCoulLongPolarization::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return NULL;
}

/* ---------------------------------------------------------------------- */

int PairLJCutCoulLongPolarization::DipoleSolverIterative()
{
  double **ef_static = atom->ef_static;
  double *static_polarizability = atom->static_polarizability;
  double **mu_induced = atom->mu_induced;
  int nlocal = atom->nlocal;
  int i,ii,j,jj,p,q,iterations,keep_iterating,index;
  double change;

  /* build dipole interaction tensor */
  build_dipole_field_matrix();

  keep_iterating = 1;
  iterations = 0;
  for(i = 0; i < nlocal; i++) ranked_array[i] = i;

  /* rank the dipoles by bubble sort */
  if(polar_gs_ranked) {
    int tmp,sorted;
    for(i = 0; i < nlocal; i++) {
      for(j = 0, sorted = 1; j < (nlocal-1); j++) {
        if(rank_metric[ranked_array[j]] < rank_metric[ranked_array[j+1]]) {
          sorted = 0;
          tmp = ranked_array[j];
          ranked_array[j] = ranked_array[j+1];
          ranked_array[j+1] = tmp;
        }
      }
      if(sorted) break;
    }
  }

  while (keep_iterating)
  {
    /* save old dipoles and clear the induced field */
    for(i = 0; i < nlocal; i++)
    {
      for(p = 0; p < 3; p++)
      {
        mu_induced_old[i][p] = mu_induced[i][p];
        ef_induced[i][p] = 0;
      }
    }

    /* contract the dipoles with the field tensor */
    for(i = 0; i < nlocal; i++) {
      index = ranked_array[i];
      ii = index*3;
      for(j = 0; j < nlocal; j++) {
        jj = j*3;
        if(index != j) {
          for(p = 0; p < 3; p++)
            for(q = 0; q < 3; q++)
              ef_induced[index][p] -= dipole_field_matrix[ii+p][jj+q]*mu_induced[j][q];
        }
      } /* end j */


      /* dipole is the sum of the static and induced parts */
      for(p = 0; p < 3; p++) {
        mu_induced_new[index][p] = static_polarizability[index]*(ef_static[index][p] + ef_induced[index][p]);

        /* Gauss-Seidel */
        if(polar_gs || polar_gs_ranked)
          mu_induced[index][p] = mu_induced_new[index][p];
      }

    } /* end i */

    double u_polar = 0.0;
    if (debug)
    {
      for (i=0;i<nlocal;i++)
      {
        u_polar += ef_static[i][0]*mu_induced[i][0] + ef_static[i][1]*mu_induced[i][1] + ef_static[i][2]*mu_induced[i][2];
      }
      u_polar *= -0.5;
      printf("u_polar (K) %d: %.18f\n",iterations,u_polar*22.432653052265*22.432653052265);
    }

    /* determine if we are done by precision */
    if (fixed_iteration==0)
    {
      keep_iterating = 0;
      change = 0;
      for(i = 0; i < nlocal; i++)
      {
        for(p = 0; p < 3; p++)
        {
          change += (mu_induced_new[i][p] - mu_induced_old[i][p])*(mu_induced_new[i][p] - mu_induced_old[i][p]);
        }
      }
      change /= (double)(nlocal)*3.0;
      if (change > polar_precision*polar_precision)
      {
        keep_iterating = 1;
      }
    }
    else
    {
      /* or by fixed iteration */
      if(iterations >= iterations_max) return iterations;
    }

    /* save the dipoles for the next pass */
    for(i = 0; i < nlocal; i++) {
      for(p = 0; p < 3; p++) {
          mu_induced[i][p] = mu_induced_new[i][p];
      }
    }

    iterations++;
    /* divergence detection */
    /* if we fail to converge, then set dipoles to alpha*E */
    if(iterations > iterations_max) {

      for(i = 0; i < nlocal; i++)
        for(p = 0; p < 3; p++)
          mu_induced[i][p] = static_polarizability[i]*ef_static[i][p];

      error->warning(FLERR,"Number of iterations exceeding max_iterations, setting dipoles to alpha*E");
      return iterations;
    }
  }
  return iterations;
}

/* ---------------------------------------------------------------------- */

/* polarization stuff */
void PairLJCutCoulLongPolarization::build_dipole_field_matrix()
{
  int N = atom->nlocal;
  double **x = atom->x;
  double *static_polarizability = atom->static_polarizability;
  int i,j,ii,jj,p,q;
  double r,r2,r3,r5,s,v,damping_term1=1.0,damping_term2=1.0;
  double xjimage[3] = {0.0,0.0,0.0};

  /* zero out the matrix */
  for (i=0;i<3*N;i++)
  {
    for (j=0;j<3*N;j++)
    {
      dipole_field_matrix[i][j] = 0;
    }
  }

  /* set the diagonal blocks */
  for(i = 0; i < N; i++) {
    ii = i*3;
    for(p = 0; p < 3; p++) {
      if(static_polarizability[i] != 0.0)
        dipole_field_matrix[ii+p][ii+p] = 1.0/static_polarizability[i];
      else
        dipole_field_matrix[ii+p][ii+p] = DBL_MAX;
    }
  }

    /* calculate each Tij tensor component for each dipole pair */
  for(i = 0; i < (N - 1); i++) {
    ii = i*3;
    for(j = (i + 1); j < N; j++) {
      jj = j*3;

      /* inverse displacements */
      double xi[3] = {x[i][0],x[i][1],x[i][2]};
      double xj[3] = {x[j][0],x[j][1],x[j][2]};
      domain->closest_image(xi,xj,xjimage);
      r2 = pow(xi[0]-xjimage[0],2)+pow(xi[1]-xjimage[1],2)+pow(xi[2]-xjimage[2],2);

      r = sqrt(r2);
      if(r == 0.0)
        r3 = r5 = DBL_MAX;
      else {
        r3 = 1.0/(r*r*r);
        r5 = 1.0/(r*r*r*r*r);
      }

      /* set the damping function */
      if(damping_type == DAMPING_EXPONENTIAL) {
        damping_term1 = 1.0 - exp(-polar_damp*r)*(0.5*polar_damp*polar_damp*r2 + polar_damp*r + 1.0);
        damping_term2 = 1.0 - exp(-polar_damp*r)*(polar_damp*polar_damp*polar_damp*r2*r/6.0 + 0.5*polar_damp*polar_damp*r2 + polar_damp*r + 1.0);
      }

      /* build the tensor */
      for(p = 0; p < 3; p++) {
        for(q = 0; q < 3; q++) {
          dipole_field_matrix[ii+p][jj+q] = -3.0*(x[i][p]-xjimage[p])*(x[i][q]-xjimage[q])*damping_term2*r5;
          /* additional diagonal term */
          if(p == q)
            dipole_field_matrix[ii+p][jj+q] += damping_term1*r3;
        }
      }

      /* set the lower half of the tensor component */
      for(p = 0; p < 3; p++)
        for(q = 0; q < 3; q++)
          dipole_field_matrix[jj+p][ii+q] = dipole_field_matrix[ii+p][jj+q];
    }
  }

  return;
}

/* ---------------------------------------------------------------------- */

int PairLJCutCoulLongPolarization::pack_comm(int n, int *list, double *buf,
           int pbc_flag, int *pbc)
{
  int i,j,m;
  double *static_polarizability = atom->static_polarizability;
  double **ef_static = atom->ef_static;
  double **mu_induced = atom->mu_induced;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = static_polarizability[j];
    buf[m++] = ef_static[j][0];
    buf[m++] = ef_static[j][1];
    buf[m++] = ef_static[j][2];
    buf[m++] = mu_induced[j][0];
    buf[m++] = mu_induced[j][1];
    buf[m++] = mu_induced[j][2];
  }
  return 7;
}

/* ---------------------------------------------------------------------- */

void PairLJCutCoulLongPolarization::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;
  double *static_polarizability = atom->static_polarizability;
  double **ef_static = atom->ef_static;
  double **mu_induced = atom->mu_induced;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    static_polarizability[i] = buf[m++];
    ef_static[i][0] = buf[m++];
    ef_static[i][1] = buf[m++];
    ef_static[i][2] = buf[m++];
    mu_induced[i][0] = buf[m++];
    mu_induced[i][1] = buf[m++];
    mu_induced[i][2] = buf[m++];
  }
}

/* end polarization stuff */

