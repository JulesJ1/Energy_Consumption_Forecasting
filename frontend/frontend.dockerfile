FROM python:3.9-slim

WORKDIR /work-dir

COPY requirements.txt ./


RUN pip install -r requirements.txt

COPY . .

WORKDIR /work-dir/frontend

EXPOSE 8050

CMD ["python","dashapp.py"]


