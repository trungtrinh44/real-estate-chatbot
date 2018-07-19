FROM python:3-slim-stretch

RUN apt update && apt upgrade -y && apt install build-essential -y
RUN pip install gunicorn
RUN pip install flask
RUN pip install flask_cors
RUN pip install numpy
RUN pip install fuzzywuzzy
RUN pip install python-Levenshtein
RUN pip install requests
RUN pip install pymongo
RUN apt purge -y build-essential && apt autoremove -y
RUN mkdir /query
COPY . query
WORKDIR /query
ENV PYTHONPATH=/query
EXPOSE 4774
ENTRYPOINT ["bash"]
CMD ["run_service.sh"]
