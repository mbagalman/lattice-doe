# Dockerfile — I-Opt Power Design Streamlit app
#
# Build:  docker build -t iopt-doe .
# Run:    docker run -p 8501:8501 iopt-doe
# Open:   http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# Copy package metadata first so dependency layer is cached separately
COPY pyproject.toml README.md ./
COPY iopt_power_design/ ./iopt_power_design/

# Install the package and all Streamlit app dependencies
RUN pip install --no-cache-dir -e ".[app,extras]"

# Copy the Streamlit app
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

# Healthcheck — Streamlit exposes /healthz
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:8501/healthz || exit 1

CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
