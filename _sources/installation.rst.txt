Installation
============

Package Installation
____________________
.. code-block:: bash

    conda create -y -n scquest python=3.8
    conda activate scquest
    pip install "git+https://github.com/AI4SCR/scQUEST.git@master"

Miniconda Installation
______________________
.. code-block:: bash

    mkdir tmp && cd tmp
    # download installation script
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod u+x Miniconda3-latest-Linux-x86_64.sh;
    bash Miniconda3-latest-Linux-x86_64.sh -b;
    # clean up
    cd .. && rm -r tmp
    # set up shell
    ~/miniconda3/bin/conda init bash
    source .bashrc