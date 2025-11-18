FROM python:3.11
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app/
EXPOSE 8502
CMD ["streamlit", "run","doctor.py"]