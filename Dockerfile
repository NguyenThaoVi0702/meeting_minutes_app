# Dockerfile

# Start from your mandatory base image which contains necessary system-level
# dependencies for diarization (e.g., CUDA, cuDNN, specific libraries).
FROM vint1-diarization:v1

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create the application directory
WORKDIR /app

# Install pip dependencies
# First, copy only the requirements file to leverage Docker's build cache.
COPY requirements.txt .

# Install all Python packages. This will add our application-specific
# libraries on top of the ones already present in the base image.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application source code into the container
# This includes the api, services, processing, worker, etc. directories.
COPY ./app /app

# Expose the port the API server will run on
EXPOSE 8000

# Define the default command to run when a container starts.
# This will start the FastAPI web server. For our workers, we will
# override this command in the docker-compose.yml file.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]