FROM conda/miniconda3
COPY . /stellar-numerics
RUN conda init bash
RUN . ~/.bashrc && conda env create -f /stellar-numerics/environment.yml
# Couldn't load asv.plugins._mamba_helpers because
# No module named 'libmambapy'
# No module named 'conda'
RUN . ~/.bashrc && conda activate stellar-numerics && conda install conda git libmambapy
# No information stored about machine '<runner name>'. I know about nothing.
# Run asv at the console the first time to generate one, or run `asv machine --yes`.
RUN . ~/.bashrc && conda activate stellar-numerics && cd /stellar-numerics && asv machine --yes
