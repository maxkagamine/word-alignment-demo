FROM python:3.12-slim

WORKDIR /app
EXPOSE 5000

COPY requirements.txt .

RUN --mount=type=cache,id=pip,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

CMD ["flask", "-A", "visualize.py", "run", "-h", "0.0.0.0"]
