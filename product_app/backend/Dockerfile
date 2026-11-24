FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY kaggle_pipeline /workspace/kaggle_pipeline
COPY product_app/backend /workspace/product_app/backend

# Si no existe un modelo exportado aún, genera el baseline reproducible automáticamente
RUN python /workspace/kaggle_pipeline/src/train.py

WORKDIR /workspace/product_app/backend
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
