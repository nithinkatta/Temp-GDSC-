FROM python:3.10.11
COPY requirements.txt /tmp/requirements.txt
# RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
COPY . /app
WORKDIR /app/
EXPOSE 5000:5000
CMD ["python","flask__.py"]