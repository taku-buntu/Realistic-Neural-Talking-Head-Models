FROM nvidia/cuda:10.1-devel-ubuntu18.04

# apt install python3.6
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN add-apt-repository ppa:jonathonf/vim
RUN apt update && apt install -y python3.6 python-dev python3.6-dev python3-pip 
RUN apt install -y vim libsm6 libxext6 libxrender-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 10
RUN pip3 install --upgrade pip
RUN apt install -y sudo
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y wget git locales
RUN locale-gen "en_US.UTF-8"
RUN update-locale LC_ALL="en_US.UTF-8"

WORKDIR /workspace
RUN apt install --yes caffe-cuda uvccapture guvcview cheese
RUN pip3 install mmdnn face-alignment==1.0.0 matplotlib==3.1.1 numpy==1.16 opencv-python==4.1.0.25 torch==1.4.0 torchvision scikit-image==0.15 scipy==1.4.1

#setting USER group number
ENV USER docker
RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER}
RUN chmod u+s /usr/sbin/useradd \
   && chmod u+s /usr/sbin/groupadd
ENV HOME /home/${USER}
ENV SHELL /bin/bash
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TERM xterm-256color
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'USER ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER}

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

WORKDIR /workspace