FROM python:3.6.6-slim

RUN mkdir /real-estate-model
ADD ./requirements.txt /real-estate-model
COPY ./data_utils /real-estate-model/data_utils
COPY ./run-prediction-service.sh /real-estate-model
WORKDIR /real-estate-model
RUN pip install --no-cache -r requirements.txt
COPY ./server /real-estate-model/server
ENV PYTHONPATH=/real-estate-model
EXPOSE 5000
ENTRYPOINT ["bash"]
CMD ["run-prediction-service.sh"]
