FROM rocker/r-ver:4.6.1

WORKDIR /app

COPY install_packages.R .
RUN Rscript install_packages.R

COPY titanic_model.R .

CMD ["Rscript", "titanic_model.R"]
