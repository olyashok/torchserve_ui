FROM python:3.9

EXPOSE 8502

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx


RUN mkdir -p /streamlit
COPY streamlit /streamlit
WORKDIR /streamlit

ENTRYPOINT [ "streamlit", "run"]
CMD ["--server.port", "8502", "torchserve.py"]