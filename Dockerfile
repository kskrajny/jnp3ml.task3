FROM ubuntu:latest
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.8 python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN mkdir src
WORKDIR /src
ADD requirements.txt /src/requirements.txt
RUN pip3 install -r requirements.txt
ADD . .
RUN pip3 install jupyter
ARG train
RUN chmod u+x /src/scripts/train.sh && /src/scripts/train.sh $train
WORKDIR /src/notebooks

CMD ["/src/entrypoint.sh"]

ENTRYPOINT  ["/bin/bash"]