FROM python:3.9

WORKDIR /app
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -U wheel cmake
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5005
USER 1001

RUN cd ./app
CMD ["python", "app.py"]
