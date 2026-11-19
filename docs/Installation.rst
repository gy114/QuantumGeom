.. _Installation:

Installation
============

QuantumGeom builds from source with CMake. The repository includes Conda
environments for Linux and macOS. On Windows, follow the Linux instructions
inside `WSL 2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

You'll need Git and a Conda distribution such as
`Miniforge <https://conda-forge.org/download/>`_. Clone the repository.

.. code-block:: bash

   git clone https://github.com/gy114/QuantumGeom.git
   cd QuantumGeom

Create the build environment
----------------------------

On Linux or WSL 2, create the Linux environment.

.. code-block:: bash

   conda env create -f environment-cmake.yml
   conda activate quantumgeom-cmake

On macOS, use Apple Clang and the macOS environment. If needed, install
`Apple's command-line tools <https://developer.apple.com/documentation/xcode/installing-the-command-line-tools>`_.

.. code-block:: bash

   conda env create -f environment-mac.yml
   conda activate quantumgeom-mac
   export CC=/usr/bin/clang CXX=/usr/bin/clang++

Build and verify
----------------

Run these commands from the repository root with the environment active.
CMake installs the extension into the selected Python's site-packages.

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
       -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
   cmake --build build --target quantumGeom --config Release --parallel 2
   cmake --install build --config Release
   python -c "import quantumGeom; print(quantumGeom.__version__)"

Try the bundled model to generate a Hall conductivity plot and CSV.

.. code-block:: bash

   python examples/weyl_hall_conductivity.py --step 2.5

Use an existing toolchain
-------------------------

You can also supply dependencies through your system package manager or
cluster modules. The Python extension needs a C++17 compiler with OpenMP,
CMake, Python 3 development headers, pybind11, CGAL 6, Armadillo, BLAS, LAPACK,
and VTK. Set ``Python3_EXECUTABLE`` to the interpreter you will use and set
``CMAKE_PREFIX_PATH`` to any custom dependency prefixes. The Weyl example
also uses NumPy and Matplotlib.
