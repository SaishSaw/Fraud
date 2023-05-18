FROM python
WORKDIR /pythondir
COPY . /pythondir
EXPOSE 8501
RUN pip install -r requirements.txt
CMD streamlit run server.py