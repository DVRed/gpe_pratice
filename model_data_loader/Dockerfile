FROM python:3.10-slim

# install latest updates
RUN apt-get update && apt-get -y install
RUN apt-get -y install libpq-dev gcc

# install latest updates
RUN pip install --upgrade pip

# set work directory
WORKDIR /model_data_loader

# copy requirements.txt to work directory and install pip's
COPY requirements.txt /model_data_loader
RUN pip install -r requirements.txt

# copy scripts to work directory
COPY srv/sstd/ /model_data_loader/srv/sstd/
COPY *.py /model_data_loader/
COPY /utils /model_data_loader/utils
COPY model_data_loader.sh /model_data_loader
RUN sed -i -e 's/\r$//' model_data_loader.sh

# set access rights
RUN chmod o+w /model_data_loader/
RUN chmod +x /model_data_loader/model_data_loader.sh

# set Europe/Moscow timezone
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# create airflow user without password
RUN adduser --disabled-password --gecos '' airflow
USER airflow

# run bash script
CMD ["bash", "model_data_loader.sh"]
