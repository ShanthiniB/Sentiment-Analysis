# Use official Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Upgrade pip and install dependencies (Flask, Gunicorn, sklearn, etc.)
RUN pip install --upgrade pip
RUN pip install flask gunicorn scikit-learn pandas numpy

# Expose the port that Flask app runs on (default 5000)
EXPOSE 5000

# Run Gunicorn server with 4 workers binding to 0.0.0.0:5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
