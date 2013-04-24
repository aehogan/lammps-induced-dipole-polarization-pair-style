# Install/unInstall package files in LAMMPS

if (test $1 = 1) then

  if (test -e ../pair_eam_alloy.cpp) then
    cp pair_cdeam.cpp ..
    cp pair_cdeam.h ..
  fi

  cp angle_cosine_shift.cpp ..
  cp angle_cosine_shift_exp.cpp ..
  cp angle_dipole.cpp ..
  cp angle_fourier.cpp ..
  cp angle_fourier_simple.cpp ..
  cp angle_quartic.cpp ..
  cp bond_harmonic_shift.cpp ..
  cp bond_harmonic_shift_cut.cpp ..
  cp compute_ackland_atom.cpp ..
  cp compute_temp_rotate.cpp ..
  cp dihedral_cosine_shift_exp.cpp ..
  cp dihedral_fourier.cpp ..
  cp dihedral_nharmonic.cpp ..
  cp dihedral_quadratic.cpp ..
  cp dihedral_table.cpp ..
  cp improper_fourier.cpp ..
  cp fix_addtorque.cpp ..
  cp fix_imd.cpp ..
  cp fix_smd.cpp ..
  cp improper_cossq.cpp ..
  cp improper_ring.cpp ..
  cp pair_coul_diel.cpp ..
  cp pair_dipole_sf.cpp ..
  cp pair_edip.cpp ..
  cp pair_gauss_cut.cpp ..
  cp pair_lj_sf.cpp ..
  cp pair_meam_spline.cpp ..
  cp pair_meam_sw_spline.cpp ..
  cp pair_tersoff_table.cpp ..

  cp angle_cosine_shift.h ..
  cp angle_cosine_shift_exp.h ..
  cp angle_dipole.h ..
  cp angle_fourier.h ..
  cp angle_fourier_simple.h ..
  cp angle_quartic.h ..
  cp bond_harmonic_shift.h ..
  cp bond_harmonic_shift_cut.h ..
  cp compute_ackland_atom.h ..
  cp compute_temp_rotate.h ..
  cp dihedral_cosine_shift_exp.h ..
  cp dihedral_fourier.h ..
  cp dihedral_nharmonic.h ..
  cp dihedral_quadratic.h ..
  cp dihedral_table.h ..
  cp improper_fourier.h ..
  cp fix_addtorque.h ..
  cp fix_imd.h ..
  cp fix_smd.h ..
  cp improper_cossq.h ..
  cp improper_ring.h ..
  cp pair_coul_diel.h ..
  cp pair_dipole_sf.h ..
  cp pair_edip.h ..
  cp pair_gauss_cut.h ..
  cp pair_lj_sf.h ..
  cp pair_meam_spline.h ..
  cp pair_meam_sw_spline.h ..
  cp pair_tersoff_table.h ..

elif (test $1 = 0) then

  rm -f ../angle_cosine_shift.cpp
  rm -f ../angle_cosine_shift_exp.cpp
  rm -f ../angle_fourier.cpp
  rm -f ../angle_fourier_simple.cpp
  rm -f ../angle_dipole.cpp
  rm -f ../angle_quartic.cpp
  rm -f ../bond_harmonic_shift.cpp
  rm -f ../bond_harmonic_shift_cut.cpp
  rm -f ../compute_ackland_atom.cpp
  rm -f ../compute_temp_rotate.cpp
  rm -f ../dihedral_cosine_shift_exp.cpp
  rm -f ../dihedral_fourier.cpp
  rm -f ../dihedral_nharmonic.cpp
  rm -f ../dihedral_quadratic.cpp
  rm -f ../dihedral_table.cpp
  rm -f ../improper_fourier.cpp
  rm -f ../fix_addtorque.cpp
  rm -f ../fix_imd.cpp
  rm -f ../fix_smd.cpp
  rm -f ../improper_cossq.cpp
  rm -f ../improper_ring.cpp
  rm -f ../pair_cdeam.cpp
  rm -f ../pair_coul_diel.cpp
  rm -f ../pair_dipole_sf.cpp
  rm -f ../pair_edip.cpp
  rm -f ../pair_gauss_cut.cpp
  rm -f ../pair_lj_sf.cpp
  rm -f ../pair_meam_spline.cpp
  rm -f ../pair_meam_sw_spline.cpp
  rm -f ../pair_tersoff_table.cpp

  rm -f ../angle_cosine_shift.h
  rm -f ../angle_cosine_shift_exp.h
  rm -f ../angle_dipole.h
  rm -f ../angle_fourier.h
  rm -f ../angle_fourier_simple.h
  rm -f ../angle_quartic.h
  rm -f ../bond_harmonic_shift.h
  rm -f ../bond_harmonic_shift_cut.h
  rm -f ../compute_ackland_atom.h
  rm -f ../compute_temp_rotate.h
  rm -f ../dihedral_cosine_shift_exp.h
  rm -f ../dihedral_fourier.h
  rm -f ../dihedral_nharmonic.h
  rm -f ../dihedral_quadratic.h
  rm -f ../dihedral_table.h
  rm -f ../improper_fourier.h
  rm -f ../fix_addtorque.h
  rm -f ../fix_imd.h
  rm -f ../fix_smd.h
  rm -f ../improper_cossq.h
  rm -f ../improper_ring.h
  rm -f ../pair_cdeam.h
  rm -f ../pair_coul_diel.h
  rm -f ../pair_dipole_sf.h
  rm -f ../pair_edip.h
  rm -f ../pair_gauss_cut.h
  rm -f ../pair_lj_sf.h
  rm -f ../pair_meam_spline.h
  rm -f ../pair_meam_sw_spline.h
  rm -f ../pair_tersoff_table.h

fi
