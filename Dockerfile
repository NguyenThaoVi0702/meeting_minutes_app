FROM dso-nexus.vietinbank.vn/ai_docker/diarization_transcription_service:v1


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN pip config set global.index-url ****
RUN pip config set global.trusted-host ****

#RUN apt-get update && apt-get install -y --no-install-recommends pandoc

WORKDIR /code

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout 60 -r requirements.txt

    
COPY ./app /code/app


EXPOSE 8072


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8072"]
