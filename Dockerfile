FROM python:3.7-slim-stretch
WORKDIR /app
COPY ./ /app/
RUN pip install -r requirements.txt --no-cache-dir
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]