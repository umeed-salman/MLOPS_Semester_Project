FROM python:3.8


WORKDIR /app


COPY . .


RUN pip install -r requirements.txt
RUN pip install dvc
RUN pip install dvc-gdrive
RUN dvc pull data


CMD ["python", "main.py"]
