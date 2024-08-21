FROM ghcr.io/infracloudio/python-langchain-huggingface:3.10-slim

WORKDIR /app

COPY .streamlit/** /app/.streamlit/ 
COPY templates/** /app/templates/
COPY app.py multi_tenant_rag.py requirements.txt tei_rerank.py /app/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8051

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
