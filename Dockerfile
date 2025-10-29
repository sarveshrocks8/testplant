# Use official Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Render uses 10000)
EXPOSE 10000

# Command to run your Flask app
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
