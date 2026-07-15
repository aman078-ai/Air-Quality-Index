# Use an official lightweight Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to optimize Python execution in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies if required (gcc is sometimes needed for compiling certain packages, but for pre-built wheels it's fast)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY generate_data.py .
COPY pipeline.py .
COPY train.py .
COPY app.py .

# Generate the synthetic dataset and train the machine learning model
# so the container runs with a pre-trained model out-of-the-box.
RUN python generate_data.py && python train.py

# Expose Streamlit's default port
EXPOSE 8501

# Add a healthcheck to ensure the container is running correctly
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the Streamlit dashboard on container startup
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
