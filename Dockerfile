FROM ubuntu:latest
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.8 python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN mkdir src
WORKDIR /src
ADD requirements.txt /src/requirements.txt
RUN pip3 install -r requirements.txt
ADD jupyter_require.txt /src/jupyter_require.txt
RUN pip3 install -r jupyter_require.txt
ADD . .
ARG train
RUN if [ $train -eq 1 ]; then python3 -u /src/main.py; fi
RUN ls

CMD ["/src/scripts/entrypoint.sh"]

ENTRYPOINT  ["/bin/bash"]