import os
import sys

from books_recommender.log_details.log import logging
from books_recommender.exception.exception_handler import AppException
from books_recommender.config.configuration import AppConfiguration


class DataIngestion:         #This class contains everything related to the data ingestion stage

# This asks your AppConfiguration:
# "Give me the configuration for Data Ingestion."
    def __init__(self, app_config=AppConfiguration()):
        """
        Data Ingestion Initialization.
        Gets the data ingestion configuration.
        """
        try:
            logging.info(
                f"{'=' * 20} Data Ingestion log started. {'=' * 20}"
            )
# So now the object knows things like:
# data/Books.csv
# data/Users.csv
# data/Book-Ratings.csv
            self.data_ingestion_config = (
                app_config.get_data_ingestion_config()
            )

        except Exception as e:
            raise AppException(e, sys) from e


    def initiate_data_ingestion(self):
        """
        Verify that the required CSV files exist
        in the data directory.
        """

        try:
            books_file = self.data_ingestion_config.books_csv_file
            users_file = self.data_ingestion_config.users_csv_file
            ratings_file = self.data_ingestion_config.ratings_csv_file

            # Check that all required files exist
            required_files = [
                books_file,
                users_file,
                ratings_file
            ]

            for file_path in required_files:

                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Required data file not found: {file_path}"
                    )

                logging.info(
                    f"Data file found: {file_path}"
                )

            logging.info(
                f"{'=' * 20} Data Ingestion log completed. {'=' * 20}\n\n"
            )

            return required_files

        except Exception as e:
            raise AppException(e, sys) from e