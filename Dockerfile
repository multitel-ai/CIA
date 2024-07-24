FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install alibi-detect==0.11.2
RUN pip install imagecorruptions==1.1.2
RUN pip install phx-class-registry==4.0.6
RUN pip install statsmodels==0.14.0
RUN pip install plotly==5.14.1
RUN pip install pydantic-numpy==2.2.1
RUN pip install pydantic-yaml
RUN pip install jupyter==1.0.0
RUN pip install nose2==0.13.0
RUN pip install pytest==7.4.0
RUN pip install pytorch-lightning==2.0.7
RUN pip install tensorboard==2.14.0
RUN pip install grad-cam==1.4.8
RUN pip install datasets==2.14.4

# DOC
RUN pip install sphinx==7.2.5
RUN pip install sphinx_autodoc_typehints==1.24.0
RUN pip install sphinxcontrib-apidoc==0.4.0
RUN pip install nbsphinx==0.9.3
RUN pip install myst-parser==2.0.0
RUN pip install sphinx_design==0.5.0
RUN pip install sphinx-rtd-theme==1.3.0

RUN pip install pylint==2.17.5

# to build doc to pdf https://www.sphinx-doc.org/en/master/usage/builders/index.html#sphinx.builders.latex.LaTeXBuilder
RUN DEBIAN_FRONTEND=noninteractive apt-get install tzdata -y
RUN apt-get install build-essential -y

# Install zsh and oh-my-zsh
RUN apt-get install curl zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# # autoML
# RUN apt-get install build-essential -y
# RUN pip install Cython==0.29.36
# RUN pip install scipy==1.9
# RUN pip install scikit-learn==0.24.2 --no-build-isolation
# RUN pip install auto-sklearn==0.15.0

RUN pip install seaborn==0.12.2
RUN pip install lmdb==1.4.1

WORKDIR gen_data

ARG GROUP_ID=1000
ARG USER_ID=1000

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user
