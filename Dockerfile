FROM python:3.10.6-buster

# WORKDIR /prod

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Then only, install taxifare!
COPY main.py main.py
COPY setup.py setup.py
RUN pip install .

COPY models models

# # We already have a make command for that!
# COPY Makefile Makefile
# RUN make reset_local_files

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
