# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock /app/

# Copy the application code before installing dependencies
COPY ./src /app/src

COPY ./poetry.lock /app/
COPY ./pyproject.toml /app/
COPY ./README.md /app/

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app/src

# Install Poetry
RUN pip install poetry

# Configure Poetry to not create virtual environments
RUN poetry config virtualenvs.create false

# Install project dependencies
RUN poetry install

# Expose the port where the FastAPI app will run
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["poetry", "run", "uvicorn", "src.faivor.api_controller:app", "--host", "0.0.0.0", "--port", "8000"]
