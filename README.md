<p align="center">peridynamics</p>

Python code for solving some 2D peridynamic problems based on Tulio Patriota's [PDLAB](https://github.com/TulioVBP/PDLAB). It was done for my mechanical engineergin college's degree and it's not complete and probably contains a few bugs here and there.

<p align="center">
    <img alt="Mesh displacement" src="https://lh3.googleusercontent.com/u/0/drive-viewer/AAOQEOQ8rSkIkdjI11dPkESGFjhb5qop6YDXR1zsfjW2O596dUKlD8pO4MyJUzKqmZVU3nCYAKrzEcMcZOqQL8LZGaVlS9FGYQ=w3946-h2680" width="45%" style=vertical-align:middle></a>
    <img alt="Displacement plot" src="https://lh3.googleusercontent.com/u/0/drive-viewer/AAOQEOSms0qgC9GhpH6mIi2IemEhR10GCc20igHFOEMHsJDYpZFQyJs8zl_qaMZI7-8HeleS7vuUt2jBoZ6MzhEJ1h1JRu55=w3726-h1797" width="45%" style=vertical-align:middle></a>
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
  <img alt="Diagram" src="https://lh3.googleusercontent.com/u/0/drive-viewer/AAOQEOSVI9ONGfJfaPeTbxC8d9AH2-EUaMigt82ghbJPx84DZT6lxW0LW_PWsVnArYowzETu_S1x4060JYUsbxRrTkYrj82fRg=w3124-h1604" width="60%"></a>
</p>

## Libraries

Consist of import the *peridynamics* library and any other needed library such as *numpy* and *matplotlib*.

## Material constants and peridynamics parameters

It's needed to specify the material Young's modulus, Poisson's ratio, density and energy release rate. The peridynamics parameters needed are: horizon, mesh ratio and two influence function parameters.

<p align="center">
  <img alt="Influence function example: cubic polynomial" src="https://lh3.googleusercontent.com/u/0/drive-viewer/AAOQEOQqcPs3aCPE6rLRJO4M3883wvwbJjbLCUqeo6_iCVpwhGld_yPbKDofCDI7VXEUxs1_r4xIDlrroAw3ybEqMOfwYYeo=w1597-h1597" width="60%"></a>
</p>

## Mesh

The mesh is an instantiated object containing the mesh points, plus with conventional attributes and methods, such as plotting. It's possible to create both rectangular and non rectangular meshes.

<p align="center">
    <img alt="Non rectangular mesh" src="https://lh3.googleusercontent.com/u/0/drive-viewer/AAOQEOQ7ymAmFPJiLQ5ZQTqJ5ppyq0UIv6aesO2qujNxBmY4eyDFE02acvZvsFbt9yR_pzPTPj5otJdvA1r68sRm-GLbgYKh=w3871-h3676" width="45%" style=vertical-align:middle></a>
    <img alt="Rectangular mesh" src="https://lh3.googleusercontent.com/u/0/drive-viewer/AAOQEOReSW5z7kv4lqb589ac9_X_jtmbKL7anuV532Q0OG7P2x3LIQE2zK79rYLInoEyEKgTCcWIvFuzlbhshftmszfRCssq7A=w3998-h1082" width="45%" style=vertical-align:middle></a>
</p>

## Family and Partial Area

The peridynamic family is also an instantiated object containing useful information regarding family and partial area. One particular interesting feature is the implementation of vectorized algorithms found in the literature, which can be much faster than standard ones!

## Boundary conditions

Here we put a combination of prescribed body forces, displacements or velocities, dependind on the application and the solver used. It's also possible to define the initial crack segment.

<p align="center">
  <img alt="Boundary Conditions" src="https://lh3.googleusercontent.com/u/0/drive-viewer/AAOQEOSnEEl7u6ayja8lmUDaUnlz482G9udmWH12uxVviiaqSzRXQYe4A9bpBnwIep3cLA56zFENGwapCBYBPjKLfOk5PAU89w=w3998-h1268" width="60%"></a>
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
