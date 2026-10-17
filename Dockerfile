FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ichimoku_hosoda ./ichimoku_hosoda
COPY README.md ichimoku_hosoda_theory.md ./
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn ichimoku_hosoda.web:app --host 0.0.0.0 --port ${PORT}"]
