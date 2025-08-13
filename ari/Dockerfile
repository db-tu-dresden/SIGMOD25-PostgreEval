FROM ubuntu:noble

EXPOSE 8888
STOPSIGNAL SIGINT

RUN apt update && apt install -y \
        build-essential sudo locales tzdata procps lsof wget \
        bison flex curl pkg-config cmake llvm clang \
        libicu-dev libreadline-dev libssl-dev liblz4-dev libossp-uuid-dev libzstd-dev zlib1g-dev \
        git vim unzip zstd default-jre tmux \
        python3 python3-venv python3-pip \
        pandoc texlive-xetex texlive-fonts-recommended texlive-plain-generic ; \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

WORKDIR /ari

ENV LANG=en_US.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV USERNAME=ari
ENV USER=ari
RUN useradd -ms /bin/bash $USERNAME ; \
    chown -R $USERNAME:$USERNAME /ari ; \
    chmod 755 /ari ; \
    echo "$USERNAME:$USERNAME" | chpasswd ; \
    usermod -aG sudo $USERNAME ; \
    echo "$USER ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/$USER
USER $USERNAME
COPY docker-init.sh /docker-init.sh

VOLUME /ari

CMD ["/docker-init.sh"]
