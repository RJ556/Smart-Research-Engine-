FROM python:3.12.3-slim

COPY . /GGG

WORKDIR /GGG

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]