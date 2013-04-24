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
   Contributing author: Pieter J. in 't Veld (SNL)
                        Adam Hogan (USF)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math_vector.h"
#include "pair_lj_long_coul_long_polarization.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "kspace.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "float.h"

using namespace LAMMPS_NS;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

enum{DAMPING_EXPONENTIAL,DAMPING_NONE};

/* ---------------------------------------------------------------------- */

PairLJLongCoulLongPolarization::PairLJLongCoulLongPolarization(LAMMPS *lmp) : Pair(lmp)
{
  /* check for possible errors */
  if (atom->static_polarizability_flag==0) error->all(FLERR,"Pair style lj/cut/coul/long/polarization requires atom attribute polarizability");
  dispersionflag = ewaldflag = pppmflag = 1;
  respa_enable = 1;
  ftable = NULL;
  qdist = 0.0;

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
}
 
/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

#define PAIR_ILLEGAL        "Illegal pair_style lj/coul command"
#define PAIR_CUTOFF        "Only one cut-off allowed when requesting all long"
#define PAIR_MISSING        "Cut-offs missing in pair_style lj/coul"
#define PAIR_COUL_CUT        "Coulombic cut not supported in pair_style lj/coul"
#define PAIR_LARGEST        "Using largest cut-off for lj/coul long long"
#define PAIR_MIX        "Mixing forced for lj coefficients"

void PairLJLongCoulLongPolarization::options(char **arg, int order)
{
  const char *option[] = {"long", "cut", "off", NULL};
  int i;

  if (!*arg) error->all(FLERR,PAIR_ILLEGAL);
  for (i=0; option[i]&&strcmp(arg[0], option[i]); ++i);
  switch (i) {
    default: error->all(FLERR,PAIR_ILLEGAL);
    case 0: ewald_order |= 1<<order; break;
    case 2: ewald_off |= 1<<order;
    case 1: break;
  }
}

void PairLJLongCoulLongPolarization::settings(int narg, char **arg)
{
  if (narg < 3) error->all(FLERR,"Illegal pair_style command");

  ewald_off = 0;
  ewald_order = 0;
  options(arg, 6);
  options(++arg, 1);
  if (!comm->me && ewald_order&(1<<6)) error->warning(FLERR,PAIR_MIX);
  if (!comm->me && ewald_order==((1<<1)|(1<<6)))
    error->warning(FLERR,PAIR_LARGEST);
  if (!*(++arg)) error->all(FLERR,PAIR_MISSING);
  if (!((ewald_order^ewald_off)&(1<<1))) error->all(FLERR,PAIR_COUL_CUT);
  cut_lj_global = force->numeric(*(arg++));
  if (*arg&&(ewald_order&0x42==0x42)) error->all(FLERR,PAIR_CUTOFF);
  if (narg == 4) cut_coul = force->numeric(*arg);
  else cut_coul = cut_lj_global;

  if (allocated) {                                        // reset explicit cuts
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }

  --arg; --arg; --arg;
  int iarg;
  iarg = 4;
  while (iarg < narg)
  {
    if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style command");
    if (strcmp("precision",arg[iarg])==0)
    {
      polar_precision = force->numeric(arg[iarg+1]);
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
      polar_damp = force->numeric(arg[iarg+1]);
    }
    else if (strcmp("max_iterations",arg[iarg])==0)
    {
      iterations_max = force->inumeric(arg[iarg+1]);
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
      polar_gamma = force->numeric(arg[iarg+1]);
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
}

/* ----------------------------------------------------------------------
   free all arrays
------------------------------------------------------------------------- */

PairLJLongCoulLongPolarization::~PairLJLongCoulLongPolarization()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj_read);
    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(epsilon_read);
    memory->destroy(epsilon);
    memory->destroy(sigma_read);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
  if (ftable) free_tables();

  /* destroy all the arrays! */
  memory->destroy(ef_induced);
  memory->destroy(mu_induced_new);
  memory->destroy(mu_induced_old);
  memory->destroy(dipole_field_matrix);
  memory->destroy(ranked_array);
  memory->destroy(rank_metric);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut_lj_read,n+1,n+1,"pair:cut_lj_read");
  memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");
  memory->create(epsilon_read,n+1,n+1,"pair:epsilon_read");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma_read,n+1,n+1,"pair:sigma_read");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");

  /* create arrays */
  int nlocal = atom->nlocal;
  memory->create(ef_induced,nlocal,3,"pair:ef_induced");
  memory->create(mu_induced_new,nlocal,3,"pair:mu_induced_new");
  memory->create(mu_induced_old,nlocal,3,"pair:mu_induced_old");
  memory->create(dipole_field_matrix,3*nlocal,3*nlocal,"pair:dipole_field_matrix");
  memory->create(ranked_array,nlocal,"pair:ranked_array");
  memory->create(rank_metric,nlocal,"pair:rank_metric");
  nlocal_old = nlocal;
}

/* ----------------------------------------------------------------------
   extract protected data from object
------------------------------------------------------------------------- */

void *PairLJLongCoulLongPolarization::extract(const char *id, int &dim)
{
  const char *ids[] = {
    "B", "sigma", "epsilon", "ewald_order", "ewald_cut", "ewald_mix",
    "cut_coul", "cut_LJ", NULL};
  void *ptrs[] = {
    lj4, sigma, epsilon, &ewald_order, &cut_coul, &mix_flag,
    &cut_coul, &cut_lj_global, NULL};
  int i;

  for (i=0; ids[i]&&strcmp(ids[i], id); ++i);
  if (i <= 2) dim = 2;
  else dim = 0;
  return ptrs[i];
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(arg[2]);
  double sigma_one = force->numeric(arg[3]);

  double cut_lj_one = cut_lj_global;
  if (narg == 5) cut_lj_one = force->numeric(arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon_read[i][j] = epsilon_one;
      sigma_read[i][j] = sigma_one;
      cut_lj_read[i][j] = cut_lj_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::init_style()
{
  const char *style1[] = 
    {"ewald", "ewald/n", "pppm", "pppm_disp", "pppm_disp/tip4p", NULL};
  const char *style6[] = {"ewald/n", "pppm_disp", "pppm_disp/tip4p", NULL};
  int i;

  // require an atom style with charge defined

  if (!atom->q_flag && (ewald_order&(1<<1)))
    error->all(FLERR,
        "Invoking coulombic in pair style lj/coul requires atom attribute q");

  // request regular or rRESPA neighbor lists

  int irequest;

  if (update->whichflag == 0 && strstr(update->integrate_style,"respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;

    if (respa == 0) irequest = neighbor->request(this);
    else if (respa == 1) {
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    } else {
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 2;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respamiddle = 1;
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    }

  } else irequest = neighbor->request(this);

  cut_coulsq = cut_coul * cut_coul;

  // set rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0)
    cut_respa = ((Respa *) update->integrate)->cutoff;
  else cut_respa = NULL;

  // ensure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->all(FLERR,"Pair style requires a KSpace style");
  if (force->kspace) g_ewald = force->kspace->g_ewald;
  if (force->kspace) g_ewald_6 = force->kspace->g_ewald_6;

  // setup force tables

  if (ncoultablebits) init_tables(cut_coul,cut_respa);
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJLongCoulLongPolarization::init_one(int i, int j)
{
  if ((ewald_order&(1<<6))||(setflag[i][j] == 0)) {
    epsilon[i][j] = mix_energy(epsilon_read[i][i],epsilon_read[j][j],
                               sigma_read[i][i],sigma_read[j][j]);
    sigma[i][j] = mix_distance(sigma_read[i][i],sigma_read[j][j]);
    if (ewald_order&(1<<6))
      cut_lj[i][j] = cut_lj_global;
    else
      cut_lj[i][j] = mix_distance(cut_lj_read[i][i],cut_lj_read[j][j]);
  }
  else {
    sigma[i][j] = sigma_read[i][j];
    epsilon[i][j] = epsilon_read[i][j];
    cut_lj[i][j] = cut_lj_read[i][j];
  }

  double cut = MAX(cut_lj[i][j], cut_coul + 2.0*qdist);
  cutsq[i][j] = cut*cut;
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  // check interior rRESPA cutoff

  if (cut_respa && MIN(cut_lj[i][j],cut_coul) < cut_respa[3])
    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  if (offset_flag) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  cutsq[j][i] = cutsq[i][j];
  cut_ljsq[j][i] = cut_ljsq[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  return cut;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon_read[i][j],sizeof(double),1,fp);
        fwrite(&sigma_read[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj_read[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::read_restart(FILE *fp)
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
          fread(&epsilon_read[i][j],sizeof(double),1,fp);
          fread(&sigma_read[i][j],sizeof(double),1,fp);
          fread(&cut_lj_read[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon_read[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma_read[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj_read[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&ncoultablebits,sizeof(int),1,fp);
  fwrite(&tabinner,sizeof(double),1,fp);
  fwrite(&ewald_order,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_lj_global,sizeof(double),1,fp);
    fread(&cut_coul,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&ncoultablebits,sizeof(int),1,fp);
    fread(&tabinner,sizeof(double),1,fp);
    fread(&ewald_order,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&ncoultablebits,1,MPI_INT,0,world);
  MPI_Bcast(&tabinner,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&ewald_order,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   compute pair interactions
------------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::compute(int eflag, int vflag)
{
/* ---- start polarization stuff ---- */
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int i,j;
  double r;
  double **x = atom->x, *x0 = x[0];

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
  double *static_polarizability = atom->static_polarizability;
  int *molecule = atom->molecule;
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
/* ---- end polarization stuff ---- */

  double evdwl,ecoul,fpair;
  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **f = atom->f, *f0 = f[0], *fi = f0;
  double *q = atom->q;
  int *type = atom->type;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  int order1 = ewald_order&(1<<1), order6 = ewald_order&(1<<6);
  int *ineigh, *ineighn, *jneigh, *jneighn, typei, typej, ni;
  double qi = 0.0, qri = 0.0;
  double *cutsqi, *cut_ljsqi, *lj1i, *lj2i, *lj3i, *lj4i, *offseti;
  double rsq, r2inv, force_coul, force_lj;
  double g2 = g_ewald_6*g_ewald_6, g6 = g2*g2*g2, g8 = g6*g2;
  vector xi, d;

  ineighn = (ineigh = list->ilist)+list->inum;

  for (; ineigh<ineighn; ++ineigh) {                        // loop over my atoms
    i = *ineigh; fi = f0+3*i;
    if (order1) qri = (qi = q[i])*qqrd2e;                // initialize constants
    offseti = offset[typei = type[i]];
    lj1i = lj1[typei]; lj2i = lj2[typei]; lj3i = lj3[typei]; lj4i = lj4[typei];
    cutsqi = cutsq[typei]; cut_ljsqi = cut_ljsq[typei];
    memcpy(xi, x0+(i+(i<<1)), sizeof(vector));
    jneighn = (jneigh = list->firstneigh[i])+list->numneigh[i];

    for (; jneigh<jneighn; ++jneigh) {                        // loop over neighbors
      j = *jneigh;
      ni = sbmask(j);
      j &= NEIGHMASK;

      { register double *xj = x0+(j+(j<<1));
        d[0] = xi[0] - xj[0];                                // pair vector
        d[1] = xi[1] - xj[1];
        d[2] = xi[2] - xj[2]; }

      if ((rsq = vec_dot(d, d)) >= cutsqi[typej = type[j]]) continue;
      r2inv = 1.0/rsq;

      if (order1 && (rsq < cut_coulsq)) {                // coulombic
        if (!ncoultablebits || rsq <= tabinnersq) {        // series real space
          register double r = sqrt(rsq), x = g_ewald*r;
          register double s = qri*q[j], t = 1.0/(1.0+EWALD_P*x);
          if (ni == 0) {
            s *= g_ewald*exp(-x*x);
            force_coul = (t *= ((((t*A5+A4)*t+A3)*t+A2)*t+A1)*s/x)+EWALD_F*s;
            if (eflag) ecoul = t;
          }
          else {                                        // special case
            r = s*(1.0-special_coul[ni])/r; s *= g_ewald*exp(-x*x);
            force_coul = (t *= ((((t*A5+A4)*t+A3)*t+A2)*t+A1)*s/x)+EWALD_F*s-r;
            if (eflag) ecoul = t-r;
          }
        }                                                // table real space
        else {
          register union_int_float_t t;
          t.f = rsq;
          register const int k = (t.i & ncoulmask)>>ncoulshiftbits;
          register double f = (rsq-rtable[k])*drtable[k], qiqj = qi*q[j];
          if (ni == 0) {
            force_coul = qiqj*(ftable[k]+f*dftable[k]);
            if (eflag) ecoul = qiqj*(etable[k]+f*detable[k]);
          }
          else {                                        // special case
            t.f = (1.0-special_coul[ni])*(ctable[k]+f*dctable[k]);
            force_coul = qiqj*(ftable[k]+f*dftable[k]-t.f);
            if (eflag) ecoul = qiqj*(etable[k]+f*detable[k]-t.f);
          }
        }
      }
      else force_coul = ecoul = 0.0;

      if (rsq < cut_ljsqi[typej]) {                        // lj
               if (order6) {                                        // long-range lj
          register double rn = r2inv*r2inv*r2inv;
          register double x2 = g2*rsq, a2 = 1.0/x2;
          x2 = a2*exp(-x2)*lj4i[typej];
          if (ni == 0) {
            force_lj =
              (rn*=rn)*lj1i[typej]-g8*(((6.0*a2+6.0)*a2+3.0)*a2+1.0)*x2*rsq;
            if (eflag)
              evdwl = rn*lj3i[typej]-g6*((a2+1.0)*a2+0.5)*x2;
          }
          else {                                        // special case
            register double f = special_lj[ni], t = rn*(1.0-f);
            force_lj = f*(rn *= rn)*lj1i[typej]-
              g8*(((6.0*a2+6.0)*a2+3.0)*a2+1.0)*x2*rsq+t*lj2i[typej];
            if (eflag)
              evdwl = f*rn*lj3i[typej]-g6*((a2+1.0)*a2+0.5)*x2+t*lj4i[typej];
          }
        }
        else {                                                // cut lj
          register double rn = r2inv*r2inv*r2inv;
          if (ni == 0) {
            force_lj = rn*(rn*lj1i[typej]-lj2i[typej]);
            if (eflag) evdwl = rn*(rn*lj3i[typej]-lj4i[typej])-offseti[typej];
          }
          else {                                        // special case
            register double f = special_lj[ni];
            force_lj = f*rn*(rn*lj1i[typej]-lj2i[typej]);
            if (eflag)
              evdwl = f * (rn*(rn*lj3i[typej]-lj4i[typej])-offseti[typej]);
          }
        }
      }
      else force_lj = evdwl = 0.0;

      fpair = (force_coul+force_lj)*r2inv;

      if (newton_pair || j < nlocal) {
        register double *fj = f0+(j+(j<<1)), f;
        fi[0] += f = d[0]*fpair; fj[0] -= f;
        fi[1] += f = d[1]*fpair; fj[1] -= f;
        fi[2] += f = d[2]*fpair; fj[2] -= f;
      }
      else {
        fi[0] += d[0]*fpair;
        fi[1] += d[1]*fpair;
        fi[2] += d[2]*fpair;
      }

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,ecoul,fpair,d[0],d[1],d[2]);
    }
  }

/* ---- start polarization stuff ---- */

  double f_shift = -1.0/(cut_coul*cut_coul); 
  double dvdrr;
  double xjimage[3];
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,rinv;

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
      rinv = sqrt(r2inv);
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
/* ---- end polarization stuff ---- */

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::compute_inner()
{
  double rsq, r2inv, force_coul = 0.0, force_lj, fpair;

  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *x0 = atom->x[0], *f0 = atom->f[0], *fi = f0, *q = atom->q;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;


  double cut_out_on = cut_respa[0];
  double cut_out_off = cut_respa[1];


  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  int *ineigh, *ineighn, *jneigh, *jneighn, typei, typej, ni;
  int i, j, order1 = (ewald_order|(ewald_off^-1))&(1<<1);
  double qri, *cut_ljsqi, *lj1i, *lj2i;
  vector xi, d;

  ineighn = (ineigh = list->ilist)+list->inum;
  for (; ineigh<ineighn; ++ineigh) {                        // loop over my atoms
    i = *ineigh; fi = f0+3*i;
    memcpy(xi, x0+(i+(i<<1)), sizeof(vector));
    cut_ljsqi = cut_ljsq[typei = type[i]];
    lj1i = lj1[typei]; lj2i = lj2[typei];
    jneighn = (jneigh = list->firstneigh[i])+list->numneigh[i];
    for (; jneigh<jneighn; ++jneigh) {                        // loop over neighbors
      j = *jneigh;
      ni = sbmask(j);
      j &= NEIGHMASK;

      { register double *xj = x0+(j+(j<<1));
        d[0] = xi[0] - xj[0];                                // pair vector
        d[1] = xi[1] - xj[1];
        d[2] = xi[2] - xj[2]; }

      if ((rsq = vec_dot(d, d)) >= cut_out_off_sq) continue;
      r2inv = 1.0/rsq;

      if (order1 && (rsq < cut_coulsq)) {                       // coulombic
        qri = qqrd2e*q[i];
        force_coul = ni == 0 ?
          qri*q[j]*sqrt(r2inv) : qri*q[j]*sqrt(r2inv)*special_coul[ni];
      }

      if (rsq < cut_ljsqi[typej = type[j]]) {                // lennard-jones
        register double rn = r2inv*r2inv*r2inv;
        force_lj = ni == 0 ?
          rn*(rn*lj1i[typej]-lj2i[typej]) :
          rn*(rn*lj1i[typej]-lj2i[typej])*special_lj[ni];
      }
      else force_lj = 0.0;

      fpair = (force_coul + force_lj) * r2inv;

      if (rsq > cut_out_on_sq) {                        // switching
        register double rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
        fpair  *= 1.0 + rsw*rsw*(2.0*rsw-3.0);
      }

      if (newton_pair || j < nlocal) {                        // force update
        register double *fj = f0+(j+(j<<1)), f;
        fi[0] += f = d[0]*fpair; fj[0] -= f;
        fi[1] += f = d[1]*fpair; fj[1] -= f;
        fi[2] += f = d[2]*fpair; fj[2] -= f;
      }
      else {
        fi[0] += d[0]*fpair;
        fi[1] += d[1]*fpair;
        fi[2] += d[2]*fpair;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::compute_middle()
{
  double rsq, r2inv, force_coul = 0.0, force_lj, fpair;

  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *x0 = atom->x[0], *f0 = atom->f[0], *fi = f0, *q = atom->q;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  double cut_in_off = cut_respa[0];
  double cut_in_on = cut_respa[1];
  double cut_out_on = cut_respa[2];
  double cut_out_off = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  int *ineigh, *ineighn, *jneigh, *jneighn, typei, typej, ni;
  int i, j, order1 = (ewald_order|(ewald_off^-1))&(1<<1);
  double qri, *cut_ljsqi, *lj1i, *lj2i;
  vector xi, d;

  ineighn = (ineigh = list->ilist)+list->inum;

  for (; ineigh<ineighn; ++ineigh) {                        // loop over my atoms
    i = *ineigh; fi = f0+3*i;
    qri = qqrd2e*q[i];
    memcpy(xi, x0+(i+(i<<1)), sizeof(vector));
    cut_ljsqi = cut_ljsq[typei = type[i]];
    lj1i = lj1[typei]; lj2i = lj2[typei];
    jneighn = (jneigh = list->firstneigh[i])+list->numneigh[i];

    for (; jneigh<jneighn; ++jneigh) {
      j = *jneigh;
      ni = sbmask(j);
      j &= NEIGHMASK;

      { register double *xj = x0+(j+(j<<1));
        d[0] = xi[0] - xj[0];                                // pair vector
        d[1] = xi[1] - xj[1];
        d[2] = xi[2] - xj[2]; }

      if ((rsq = vec_dot(d, d)) >= cut_out_off_sq) continue;
      if (rsq <= cut_in_off_sq) continue;
      r2inv = 1.0/rsq;

      if (order1 && (rsq < cut_coulsq))                        // coulombic
        force_coul = ni == 0 ?
          qri*q[j]*sqrt(r2inv) : qri*q[j]*sqrt(r2inv)*special_coul[ni];

      if (rsq < cut_ljsqi[typej = type[j]]) {                // lennard-jones
        register double rn = r2inv*r2inv*r2inv;
        force_lj = ni == 0 ?
          rn*(rn*lj1i[typej]-lj2i[typej]) :
          rn*(rn*lj1i[typej]-lj2i[typej])*special_lj[ni];
      }
      else force_lj = 0.0;

      fpair = (force_coul + force_lj) * r2inv;

      if (rsq < cut_in_on_sq) {                                // switching
        register double rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
        fpair  *= rsw*rsw*(3.0 - 2.0*rsw);
      }
      if (rsq > cut_out_on_sq) {
        register double rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
        fpair  *= 1.0 + rsw*rsw*(2.0*rsw-3.0);
      }

      if (newton_pair || j < nlocal) {                        // force update
        register double *fj = f0+(j+(j<<1)), f;
        fi[0] += f = d[0]*fpair; fj[0] -= f;
        fi[1] += f = d[1]*fpair; fj[1] -= f;
        fi[2] += f = d[2]*fpair; fj[2] -= f;
      }
      else {
        fi[0] += d[0]*fpair;
        fi[1] += d[1]*fpair;
        fi[2] += d[2]*fpair;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJLongCoulLongPolarization::compute_outer(int eflag, int vflag)
{
  double evdwl,ecoul,fpair;
  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x, *x0 = x[0];
  double **f = atom->f, *f0 = f[0], *fi = f0;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  int i, j, order1 = ewald_order&(1<<1), order6 = ewald_order&(1<<6);
  int *ineigh, *ineighn, *jneigh, *jneighn, typei, typej, ni, respa_flag;
  double qi = 0.0, qri = 0.0;
  double *cutsqi, *cut_ljsqi, *lj1i, *lj2i, *lj3i, *lj4i, *offseti;
  double rsq, r2inv, force_coul, force_lj;
  double g2 = g_ewald_6*g_ewald_6, g6 = g2*g2*g2, g8 = g6*g2;
  double respa_lj = 0.0, respa_coul = 0.0, frespa = 0.0;
  vector xi, d;

  double cut_in_off = cut_respa[2];
  double cut_in_on = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;

  ineighn = (ineigh = list->ilist)+list->inum;

  for (; ineigh<ineighn; ++ineigh) {                        // loop over my atoms
    i = *ineigh; fi = f0+3*i;
    if (order1) qri = (qi = q[i])*qqrd2e;                // initialize constants
    offseti = offset[typei = type[i]];
    lj1i = lj1[typei]; lj2i = lj2[typei]; lj3i = lj3[typei]; lj4i = lj4[typei];
    cutsqi = cutsq[typei]; cut_ljsqi = cut_ljsq[typei];
    memcpy(xi, x0+(i+(i<<1)), sizeof(vector));
    jneighn = (jneigh = list->firstneigh[i])+list->numneigh[i];

    for (; jneigh<jneighn; ++jneigh) {                        // loop over neighbors
      j = *jneigh;
      ni = sbmask(j);
      j &= NEIGHMASK;

      { register double *xj = x0+(j+(j<<1));
        d[0] = xi[0] - xj[0];                                // pair vector
        d[1] = xi[1] - xj[1];
        d[2] = xi[2] - xj[2]; }

      if ((rsq = vec_dot(d, d)) >= cutsqi[typej = type[j]]) continue;
      r2inv = 1.0/rsq;

      frespa = 1.0;                                       // check whether and how to compute respa corrections
      respa_flag = rsq < cut_in_on_sq ? 1 : 0;
      if (respa_flag && (rsq > cut_in_off_sq)) {
        register double rsw = (sqrt(rsq)-cut_in_off)/cut_in_diff;
        frespa = rsw*rsw*(3.0-2.0*rsw);
      }

      if (order1 && (rsq < cut_coulsq)) {                // coulombic
        if (!ncoultablebits || rsq <= tabinnersq) {        // series real space
          register double r = sqrt(rsq), s = qri*q[j];
          if (respa_flag)                                // correct for respa
            respa_coul = ni == 0 ? frespa*s/r : frespa*s/r*special_coul[ni];
          register double x = g_ewald*r, t = 1.0/(1.0+EWALD_P*x);
          if (ni == 0) {
            s *= g_ewald*exp(-x*x);
            force_coul = (t *= ((((t*A5+A4)*t+A3)*t+A2)*t+A1)*s/x)+EWALD_F*s;
            if (eflag) ecoul = t;
          }
          else {                                        // correct for special
            r = s*(1.0-special_coul[ni])/r; s *= g_ewald*exp(-x*x);
            force_coul = (t *= ((((t*A5+A4)*t+A3)*t+A2)*t+A1)*s/x)+EWALD_F*s-r;
            if (eflag) ecoul = t-r;
          }
        }                                                // table real space
        else {
          if (respa_flag) respa_coul = ni == 0 ?        // correct for respa
              frespa*qri*q[j]/sqrt(rsq) :
              frespa*qri*q[j]/sqrt(rsq)*special_coul[ni];
          register union_int_float_t t;
          t.f = rsq;
          register const int k = (t.i & ncoulmask) >> ncoulshiftbits;
          register double f = (rsq-rtable[k])*drtable[k], qiqj = qi*q[j];
          if (ni == 0) {
            force_coul = qiqj*(ftable[k]+f*dftable[k]);
            if (eflag) ecoul = qiqj*(etable[k]+f*detable[k]);
          }
          else {                                        // correct for special
            t.f = (1.0-special_coul[ni])*(ctable[k]+f*dctable[k]);
            force_coul = qiqj*(ftable[k]+f*dftable[k]-t.f);
            if (eflag) ecoul = qiqj*(etable[k]+f*detable[k]-t.f);
          }
        }
      }
      else force_coul = respa_coul = ecoul = 0.0;

      if (rsq < cut_ljsqi[typej]) {                        // lennard-jones
        register double rn = r2inv*r2inv*r2inv;
        if (respa_flag) respa_lj = ni == 0 ?                 // correct for respa
            frespa*rn*(rn*lj1i[typej]-lj2i[typej]) :
            frespa*rn*(rn*lj1i[typej]-lj2i[typej])*special_lj[ni];
        if (order6) {                                        // long-range form
          register double x2 = g2*rsq, a2 = 1.0/x2;
          x2 = a2*exp(-x2)*lj4i[typej];
          if (ni == 0) {
            force_lj =
              (rn*=rn)*lj1i[typej]-g8*(((6.0*a2+6.0)*a2+3.0)*a2+1.0)*x2*rsq;
            if (eflag) evdwl = rn*lj3i[typej]-g6*((a2+1.0)*a2+0.5)*x2;
          }
          else {                                        // correct for special
            register double f = special_lj[ni], t = rn*(1.0-f);
            force_lj = f*(rn *= rn)*lj1i[typej]-
              g8*(((6.0*a2+6.0)*a2+3.0)*a2+1.0)*x2*rsq+t*lj2i[typej];
            if (eflag)
              evdwl = f*rn*lj3i[typej]-g6*((a2+1.0)*a2+0.5)*x2+t*lj4i[typej];
          }
        }
        else {                                                // cut form
          if (ni == 0) {
            force_lj = rn*(rn*lj1i[typej]-lj2i[typej]);
            if (eflag) evdwl = rn*(rn*lj3i[typej]-lj4i[typej])-offseti[typej];
          }
          else {                                        // correct for special
            register double f = special_lj[ni];
            force_lj = f*rn*(rn*lj1i[typej]-lj2i[typej]);
            if (eflag)
              evdwl = f*(rn*(rn*lj3i[typej]-lj4i[typej])-offseti[typej]);
          }
        }
      }
      else force_lj = respa_lj = evdwl = 0.0;

      fpair = (force_coul+force_lj)*r2inv;
      frespa = respa_flag == 0 ? fpair :  fpair-(respa_coul+respa_lj)*r2inv;

      if (newton_pair || j < nlocal) {
        register double *fj = f0+(j+(j<<1)), f;
        fi[0] += f = d[0]*frespa; fj[0] -= f;
        fi[1] += f = d[1]*frespa; fj[1] -= f;
        fi[2] += f = d[2]*frespa; fj[2] -= f;
      }
      else {
        fi[0] += d[0]*frespa;
        fi[1] += d[1]*frespa;
        fi[2] += d[2]*frespa;
      }

      if (evflag) ev_tally(i,j,nlocal,newton_pair,
                           evdwl,ecoul,fpair,d[0],d[1],d[2]);
    }
  }
}

/* ---------------------------------------------------------------------- */

double PairLJLongCoulLongPolarization::single(int i, int j, int itype, int jtype,
                          double rsq, double factor_coul, double factor_lj,
                          double &fforce)
{
  double r2inv, r6inv, force_coul, force_lj;
  double g2 = g_ewald_6*g_ewald_6, g6 = g2*g2*g2, g8 = g6*g2, *q = atom->q;

  double eng = 0.0;

  r2inv = 1.0/rsq;
  if ((ewald_order&2) && (rsq < cut_coulsq)) {                // coulombic
    if (!ncoultablebits || rsq <= tabinnersq) {                // series real space
      register double r = sqrt(rsq), x = g_ewald*r;
      register double s = force->qqrd2e*q[i]*q[j], t = 1.0/(1.0+EWALD_P*x);
      r = s*(1.0-factor_coul)/r; s *= g_ewald*exp(-x*x);
      force_coul = (t *= ((((t*A5+A4)*t+A3)*t+A2)*t+A1)*s/x)+EWALD_F*s-r;
      eng += t-r;
    }
    else {                                                // table real space
      register union_int_float_t t;
      t.f = rsq;
      register const int k = (t.i & ncoulmask) >> ncoulshiftbits;
      register double f = (rsq-rtable[k])*drtable[k], qiqj = q[i]*q[j];
      t.f = (1.0-factor_coul)*(ctable[k]+f*dctable[k]);
      force_coul = qiqj*(ftable[k]+f*dftable[k]-t.f);
      eng += qiqj*(etable[k]+f*detable[k]-t.f);
    }
  } else force_coul = 0.0;

  if (rsq < cut_ljsq[itype][jtype]) {                        // lennard-jones
    r6inv = r2inv*r2inv*r2inv;
    if (ewald_order&64) {                                // long-range
      register double x2 = g2*rsq, a2 = 1.0/x2, t = r6inv*(1.0-factor_lj);
      x2 = a2*exp(-x2)*lj4[itype][jtype];
      force_lj = factor_lj*(r6inv *= r6inv)*lj1[itype][jtype]-
               g8*(((6.0*a2+6.0)*a2+3.0)*a2+a2)*x2*rsq+t*lj2[itype][jtype];
      eng += factor_lj*r6inv*lj3[itype][jtype]-
        g6*((a2+1.0)*a2+0.5)*x2+t*lj4[itype][jtype];
    }
    else {                                                // cut
      force_lj = factor_lj*r6inv*(lj1[itype][jtype]*r6inv-lj2[itype][jtype]);
      eng += factor_lj*(r6inv*(r6inv*lj3[itype][jtype]-
                               lj4[itype][jtype])-offset[itype][jtype]);
    }
  } else force_lj = 0.0;

  fforce = (force_coul+force_lj)*r2inv;
  return eng;
}

/* ---------------------------------------------------------------------- */

int PairLJLongCoulLongPolarization::DipoleSolverIterative()
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

void PairLJLongCoulLongPolarization::build_dipole_field_matrix()
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
