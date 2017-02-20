FROM python:3.6.0

COPY requirements.txt .
RUN pip3 install -r requirements.txt
