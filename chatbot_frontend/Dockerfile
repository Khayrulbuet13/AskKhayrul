FROM python:3.11-slim

WORKDIR /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

COPY ./src/ /app
COPY ./image/ /app/image

CMD ["sh", "entrypoint.sh"]