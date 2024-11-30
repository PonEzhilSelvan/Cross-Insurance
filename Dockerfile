FROM python:3.8.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

#ENTRYPOINT ["uvicorn main:app --reload --port 5000", "model_app:app", "--host", "0.0.0.0", "--port", "80"]
ENTRYPOINT ["streamlit", "run", "docker run -p.py", "--server.port=80", "--server.address=0.0.0.0"]
# docker build .
#docker run -d  -p 80:80 app:tag
#docker images ==> to check the image status
#docker ps ==> to check the container status