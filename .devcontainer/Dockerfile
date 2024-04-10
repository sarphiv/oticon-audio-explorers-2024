# Setup environment basics
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /workspace

ENV PYTHONUNBUFFERED=True


# Install system packages
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    sudo \
    nvtop \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    ninja-build && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*


# Set up user
ARG USERNAME=non-root
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN sudo chown -R $USERNAME:$USERNAME /workspace


# Set up environment
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba --strip-components=1 && \
    mkdir -p ~/.local/bin && \
    mv micromamba ~/.local/bin && \
    printf "channels:\n  - conda-forge" > ~/.mambarc && \
    micromamba create -n env -y python=3.11

# NOTE: Only installing packages for container recreation performance reasons. The project is installed on create
COPY requirements.txt .
COPY requirements.dev.txt .
RUN --mount=type=cache,target=$HOME/.cache/pip \
    eval "$(micromamba shell hook -s posix)" && \
    micromamba activate env && \
    pip install -r requirements.txt && \
    pip install -r requirements.dev.txt
