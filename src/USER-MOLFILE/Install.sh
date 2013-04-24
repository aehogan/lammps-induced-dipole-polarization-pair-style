# Install/unInstall package files in LAMMPS

if (test $1 = 1) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*molfile[^ \t]* //' ../Makefile.package
    sed -i -e 's|^PKG_SYSINC =[ \t]*|&$(molfile_SYSINC) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSLIB =[ \t]*|&$(molfile_SYSLIB) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSPATH =[ \t]*|&$(molfile_SYSPATH) |' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^include.*USER-MOLFILE.*$/d' ../Makefile.package.settings
    # multiline form needed for BSD sed on Macs
    sed -i -e '4 i \
include ..\/USER-MOLFILE\/Makefile.lammps
' ../Makefile.package.settings
  fi

  cp molfile_interface.cpp ..
  cp dump_molfile.cpp ..
  cp reader_molfile.cpp ..

  cp molfile_interface.h ..
  cp dump_molfile.h ..
  cp reader_molfile.h ..

  cp molfile_plugin.h ..
  cp vmdplugin.h ..

elif (test $1 = 0) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*molfile[^ \t]* //' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^include.*USER-MOLFILE.*$/d' ../Makefile.package.settings
  fi

  rm -f ../molfile_interface.cpp
  rm -f ../dump_molfile.cpp
  rm -f ../reader_molfile.cpp

  rm -f ../molfile_interface.h
  rm -f ../dump_molfile.h
  rm -f ../reader_molfile.h

  rm -f ../molfile_plugin.h
  rm -f ../vmdplugin.h
fi
