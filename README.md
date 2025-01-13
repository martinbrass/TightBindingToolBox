# TightBindingToolBox

[![Build Status](https://github.com/martinbrass/TightBindingToolBox/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/martinbrass/TightBindingToolBox/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/martinbrass/TightBindingToolBox/branch/main/graph/badge.svg)](https://codecov.io/gh/martinbrass/TightBindingToolBox.jl)

This module is written to calculate topological properties of tight binding models. It includes an interface to import tight binding Hamiltonians from FPLO and Wannier90. A documentation will follow hopefully soon. If you use it, please cite:

Martin Braß, Liang Si, and Karsten Held
[Phys. Rev. B 109, 085103 (2024)](https://doi.org/10.1103/PhysRevB.109.085103)

Installation
------------
* To install the package for usage run:
  
  ```sh
  julia -e 'import Pkg; Pkg.add(url="path/to/repo")'
  ```
  
* To install the package for development clone the repository and run:
  
  ```sh
  julia -e 'import Pkg; Pkg.develop(url="path/to/repo")'
  ```
  



