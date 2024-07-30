FROM python:3.9

WORKDIR /work-dir

COPY requirements.txt ./

RUN pip install -r requirements.txt

ENV ENTSOE_API_KEY = a0398bcf-3594-4daf-9908-fdbfe3c7953f

COPY . .

WORKDIR /work-dir/scripts

EXPOSE 8000

CMD uvicorn XGBoostAPI:app --host 0.0.0.0 --port 8000
