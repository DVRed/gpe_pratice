FROM python:3.10-slim

# set proxy for OS updates
COPY 00aptproxy /etc/apt/apt.conf.d

# install latest updates
RUN apt-get update && apt-get -y install
RUN apt-get -y install libpq-dev gcc

# set proxy for Python updates
ENV HTTP_PROXY="http://10.18.0.15:8080"
ENV HTTPS_PROXY="http://10.18.0.15:8080"

# install latest updates
RUN pip install --upgrade pip

# set proxy exclusion for docker sock
# docker-proxy - service of airflow docker-compose.yml
ENV NO_PROXY=".docker-proxy"

RUN mkdir /srv/sstd/

# set work directory
WORKDIR /snam

# copy requirements.txt to work directory and install pip's
COPY requirements.txt /snam
RUN pip install -r requirements.txt

# copy scripts to work directory
COPY *.py /snam/
COPY snam_parser.sh /snam
RUN sed -i -e 's/\r$//' snam_parser.sh

# set access rights
RUN chmod o+w /snam/
RUN chmod o+w /tmp
RUN chmod +x /snam/snam_parser.sh
RUN chmod -R o+w /srv/

# set Europe/Moscow timezone
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# create airflow user without password
RUN adduser --disabled-password --gecos '' airflow
USER airflow

# run bash script
CMD ["bash", "snam_parser.sh"]

