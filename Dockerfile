FROM quay.io/jupyter/minimal-notebook:python-3.12

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda update --quiet --file /tmp/conda-linux-64.lock \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

RUN pip install deepchecks==0.19.1

USER root

# install lmodern to render Quarto PDF
RUN sudo apt update \
    && sudo apt install -y lmodern

USER $NB_UID
