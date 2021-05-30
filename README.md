![SeismoVLAB Logo](Logo.png)

**Seismo-VLAB** (a.k.a **SVL**) is a simple, fast, and extendable C++ finite element software designed to optimize meso-scale simulations of linear and nonlinear wave-propagation and soil-structure interaction. **SVL** is intended not only to be used by researchers in structural and geothecnical engineering, but also in industries, laboratories, universities, etc.

* Official website: http://www.seismovlab.com
* GitHub repository: https://github.com/SeismoVLAB/SVL
* Official documentation: http://www.seismovlab.com/documentation/index.html

What can Seismo-VLAB do?
------------------------
With **Seismo-VLAB** you can solve:

* Linear and Nonlinear wave propagation problems in shallow crust.
* Linear and Nonlinear soil-structure interaction problems.
* Standard mechanics-based nonlinear structural dynamic problems.

Visit the gallery for examples: http://www.seismovlab.com/gallery.html

Installing Seismo-VLAB
----------------------
Installation of **Seismo-VLab** (Pre-Process) on Linux/MacOS/Windows requires `python3` and the following libraries:

* Numpy
* Scipy
* Matplotlib
* JSON

Installation of **Seismo-VLab** (Run-Process) on Linux/MacOS/Windows requires to download `Eigen` C++ library, `MUMPS` Library, and `PETSc` Library.

* The **Eigen C++ library** can be downloaded from the website http://eigen.tuxfamily.org/. This package needs to be unzip and its content move (for instance) to `/usr/include/eigen`. 
* The **MUMPS library** can be downloaded from the website http://mumps.enseeiht.fr/. This package needs to be unzip and compiled (for instance) at `/usr/include/mumps`.
* The **Pestc Library** library can be downloaded at the website https://www.mcs.anl.gov/petsc/. This package needs to be unzip and compiled (for instance) at `/usr/include/petsc`.

Assuming the previous libraries are successfully installed, then modify the `Makefile.inc` file such the previous path point to the right libraries:

```makefile
EIGEN_DIR = /usr/include/eigen
PETSC_DIR = /usr/include/petsc
MUMPS_DIR = /usr/include/mumps
```

Also, make sure that libraries such as: *libscalapack-openmpi*, *libblacs-openmpi*, *liblapack*, *libblas*, and *libparmetis*, *libmetis*, *libptscotch*, *libptscotcherr* are also installed.

Finally, write in terminal:
```bash
make -s DEBUG=False
```
A detailed explanation on how to install **SVL** on Windows, MacOS, and Linux can be found at:\n
http://seismovlab.com/documentation/linkInstallation.html

License
=======

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Seismo-VLAB** is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
**Seismo-VLAB** is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details http://www.gnu.org/licenses.

<!---
Citation
========
To cite Seismo-VLAB, please use :

Kusanovic Danilo, Seylabi Elnaz, Kottke Albert, and Asimaki Domniki (2020). Seismo-Vlab: A parallel object-oriented platform for reliable nonlinear seismic wave propagation and soil-structure interaction simulation. *Computers and Geotechnics*. [![DOI](https://img.shields.io/badge/DOI-10.1016/j.cma.2009.08.016-green.svg)](https://doi.org/10.1016/j.cma.2009.08.016)

```
@article{Kusanovic2020SeismoVLab,
title   = {Seismo-VLAB: A parallel object-oriented platform for reliable nonlinear seismic wave propagation and soil-structure interaction simulation.},
author  = {Kusanovic Danilo and Seylabi Elnaz and Kottke Albert and Asimaki Domniki},
journal = {To be submitted to Computer Methods in Applied Mechanics and Engineering},
volume  = {},
number  = {},
pages   = {},
year    = {2020},
issn    = {},
doi     = {},
url     = {}
}
```

Kusanovic Danilo, Seylabi Elnaz, and Asimaki Domniki (2021). Seismo-VLAB: A parallel C++ finite element software for structural and soil mechanics. *The Journal of Open Source Software*. [![DOI](https://img.shields.io/badge/DOI-10.1016/j.cma.2009.08.016-green.svg)](https://doi.org/10.1016/j.cma.2009.08.016)

```
@article{Kusanovic2021SeismoVLab,
title   = {Seismo-VLAB: A parallel C++ finite element software for structural and soil mechanics.},
author  = {Kusanovic Danilo and Seylabi Elnaz and Asimaki Domniki},
journal = {To be submitted to SoftwareX},
volume  = {},
number  = {},
pages   = {},
year    = {2021},
issn    = {},
doi     = {},
url     = {}
}
```
--->
