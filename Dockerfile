FROM python:3.5

ENV HOME /root

RUN apt-get update && apt-get -yq install gcc build-essential gfortran g++ zlib1g-dev libatlas-base-dev wget

RUN pip install numpy
RUN pip install pandas
RUN pip install scipy
RUN pip install scikit-learn

WORKDIR /src

ENTRYPOINT ["python", "-u", "main.py"]

