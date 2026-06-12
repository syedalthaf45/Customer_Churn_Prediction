FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the pipeline to generate model artifacts before serving
RUN python run_pipeline.py

# Expose port
EXPOSE 5000

# Start Flask with gunicorn (production-grade)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
