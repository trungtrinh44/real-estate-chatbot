env PYTHONUNBUFFERED=true gunicorn \
    --workers 2 \
    --timeout 600 \
    query_analyzer:app -b 0.0.0.0:4774