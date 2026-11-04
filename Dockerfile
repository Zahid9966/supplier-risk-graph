FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SRG_DATA=/app/data SRG_REPORTS=/app/reports
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .
COPY dashboard ./dashboard
COPY data/external ./data/external
COPY scripts ./scripts
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh && useradd -m app && mkdir -p data reports && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
ENTRYPOINT ["./docker-entrypoint.sh"]
