FROM python:3.11-slim

WORKDIR /app

# Install system build tools (for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy everything into container
COPY . /app/

# Set PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Start JupyterLab
CMD ["jupyter", "lab", "--notebook-dir=/app", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# # If main.py is your script, you can override this via docker-compose
# CMD ["python", "main.py"]

