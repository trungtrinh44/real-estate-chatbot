FROM python:3

RUN mkdir /real-estate-model
COPY ./data_utils /real-estate-model/data_utils
COPY ./server /real-estate-model/server
COPY ./requirements.txt /real-estate-model
WORKDIR /real-estate-model
COPY ./run-prediction-service.sh /real-estate-model
RUN pip install -r requirements.txt
ENV PYTHONPATH=/real-estate-model
EXPOSE 5000
ENTRYPOINT ["bash"]
CMD ["run-prediction-service.sh"]
