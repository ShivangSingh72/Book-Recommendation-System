import os
import sys
import pickle
import pandas as pd

from books_recommender.log_details.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException


class DataTransformation:

    def __init__(self, app_config=AppConfiguration()):
        try:
            self.data_transformation_config = app_config.get_data_transformation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    def get_data_transformer(self):
        try:
            df = pd.read_csv(self.data_transformation_config.clean_data_file)

            book_pivot = df.pivot_table(
                index="title",
                columns="user_id",
                values="rating"
            )

            book_pivot.fillna(0, inplace=True)

            book_names = book_pivot.index
            final_rating = df

            os.makedirs(
                os.path.dirname(self.data_transformation_config.book_pivot_file),
                exist_ok=True
            )

            pickle.dump(
                book_pivot,
                open(self.data_transformation_config.book_pivot_file, "wb")
            )

            pickle.dump(
                book_names,
                open(self.data_transformation_config.book_names_file, "wb")
            )

            pickle.dump(
                final_rating,
                open(self.data_transformation_config.final_rating_file, "wb")
            )

            logging.info("Transformation files saved successfully.")

        except Exception as e:
            raise AppException(e, sys) from e


    def initiate_data_transformation(self):
        try:
            logging.info(f"{'='*20}Data Transformation started.{'='*20}")

            self.get_data_transformer()

            logging.info(f"{'='*20}Data Transformation completed.{'='*20}\n\n")

        except Exception as e:
            raise AppException(e, sys) from e