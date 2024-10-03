# chatbot_api/Dockerfile

FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt 

# Copy the source code to /app in the container
COPY ./src /app/
COPY ./pdfs /app/pdfs

# Expose port 8000 for the API
EXPOSE 8000

# Set the entrypoint script to run
CMD ["sh", "/app/entrypoint.sh"]