FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim


RUN pip install -U pip
RUN pip install pipenv


WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "fhv_tripdata_2021-04.parquet", "./"]

RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

EXPOSE 9696

CMD ["python", "starter.py"]