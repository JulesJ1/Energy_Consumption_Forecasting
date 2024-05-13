FROM python:3.9-slim

WORKDIR /work-dir

COPY . /work-dir

RUN pip install -r requirements.txt

EXPOSE 8050

CMD ["python","dashapp.py"]


