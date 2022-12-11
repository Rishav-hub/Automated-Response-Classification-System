FROM python:3.8-slim-buster
RUN apt update -y && apt install awscli -y
COPY . /sentiment_analysis
WORKDIR /sentiment_analysis
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python","app.py" ]