# Base image with Python 3.9
FROM python:3.9-slim

# Set environment variables
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files to the container
COPY . /app/

# Expose the Flask port
EXPOSE 5000

# Run the Python application
CMD ["python", "final.py"]
