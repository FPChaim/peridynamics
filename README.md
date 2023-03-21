<p align="center">peridynamics</p>

Python code for solving some 2D peridynamic problems based on Tulio Patriota's [PDLAB](https://github.com/TulioVBP/PDLAB). It was done for my mechanical engineergin college's degree and it's not complete and probably contains a few bugs here and there.

<p align="center">
  <img alt="Mesh displacement" src="https://drive.google.com/uc?export=view&id=1dBVF3lN_ycHI779PncKD0NapPyDFknlF" width="45%" style=vertical-align:middle></a>
  <img alt="Displacement plot" src="https://drive.google.com/uc?export=view&id=1dBbBI_FBvdIrM2jXz8ev-I5i5qlQ8XQF" width="45%" style=vertical-align:middle></a>
</p>

![Language](https://img.shields.io/github/languages/top/FPChaim/peridynamics?color=blue)
![License](https://img.shields.io/github/license/FPChaim/peridynamics?color=g)

# Requirements

- Python 3.10 or newer;
- numpy;
- scipy;
- matplotlib.

# Instalation

- Clone this repository with any git tool, such as [GitHub Desktop](https://desktop.github.com/);
- Make sure the *peridynamics* folder is in your IDE search path or in the same folder as your current working directory;
- Test it by importing the *peridynamics* module by typing:

        import peridynamics as pd
- If there are no errors, the instalation should be complete.

# Usage

One can follow the following diagram when using the code. For a more in-depth explanation, please see examples in the folder [notebooks](notebooks).

<p align="center">
  <img alt="Diagram" src="https://drive.google.com/uc?export=view&id=1d9WzpWQ1i7MAjVk0RdOxzlrubNwkXKei" width="60%"></a>
</p>

## Libraries

Consist of importing the *peridynamics* library and any other needed library such as *numpy* and *matplotlib*.

## Material constants and peridynamics parameters

It's needed to specify the material Young's modulus, Poisson's ratio, density and energy release rate. The peridynamics parameters needed are: horizon, mesh ratio and two influence function parameters.

<p align="center">
  <img alt="Influence function example: cubic polynomial" src="https://drive.google.com/uc?export=view&id=1dG1Ms3TaKX0rauCSWDwmjU61_MvccEl3" width="60%"></a>
</p>

## Mesh

The mesh is an instantiated object containing the mesh points, plus with conventional attributes and methods, such as plotting. It's possible to create both rectangular and non rectangular meshes.

<p align="center">
  <img alt="Rectangular mesh" src="https://drive.google.com/uc?export=view&id=1dFLh3SLez50fPNlY7lpI7ZzN9DUlfUMt" width="45%" style=vertical-align:middle></a>
  <img alt="Non rectangular mesh" src="https://drive.google.com/uc?export=view&id=1dFekgeq9XYy5WBlCuL2k13lMygJaf90w" width="45%" style=vertical-align:middle></a>
</p>

## Family and Partial Area

The peridynamic family is also an instantiated object containing useful information regarding family and partial area. One particular interesting feature is the implementation of vectorized algorithms found in the literature, which can be much faster than standard ones!

## Boundary conditions

Here we put a combination of prescribed body forces, displacements or velocities, dependind on the application and the solver used. It's also possible to define the initial crack segment.

<p align="center">
  <img alt="Boundary Conditions" src="https://drive.google.com/uc?export=view&id=1dEkTWAVuCIuQP5npY-ZbNsv3Vw9dlQxb" width="60%"></a>
</p>

## Model

It's necessary to choose a peridynamic model used in the calculations considerations.

## Solver

It's possible to choose between a quasi-static solver and a dynamic one.

## Model specific calculations

Requires no user input, but most of the calculations happens there.

## Results

It's possible to generate 2D and 3D graphics with the calculated numbers and also save the resulting images using a convenient function and also save objects using the provided pickle based function (warning: pickle might containg vulnerabilites and it's only recomended to save and open your own files).

- Displacement;
- Strain;
- Energy;
- Damage index;
- Track crack.
