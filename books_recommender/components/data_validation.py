import os
import sys
import pandas as pd

from books_recommender.log_details.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException


class DataValidation:

    def __init__(self, app_config=AppConfiguration()):

        try:

            logging.info(
                f"{'=' * 20} Data Validation log started. {'=' * 20}"
            )

            self.data_validation_config = (
                app_config.get_data_validation_config()
            )

        except Exception as e:
            raise AppException(e, sys) from e


    def preprocess_data(self):

        try:

            # Reading the three input CSV files

            ratings = pd.read_csv(
                self.data_validation_config.ratings_csv_file,
                sep=";",
                encoding="latin-1"
            )

            books = pd.read_csv(
                self.data_validation_config.books_csv_file,
                sep=";",
                encoding="latin-1"
            )

            users = pd.read_csv(
                self.data_validation_config.users_csv_file,
                sep=";",
                encoding="latin-1"
            )

            logging.info(
                f"Shape of ratings data: {ratings.shape}"
            )

            logging.info(
                f"Shape of books data: {books.shape}"
            )

            logging.info(
                f"Shape of users data: {users.shape}"
            )


            # ------------------------------------------------
            # BOOK DATA
            # ------------------------------------------------

            books = books[
                [
                    "ISBN",
                    "Book-Title",
                    "Book-Author",
                    "Year-Of-Publication",
                    "Publisher",
                    "Image-URL-L"
                ]
            ]

            books.rename(
                columns={
                    "Book-Title": "title",
                    "Book-Author": "author",
                    "Year-Of-Publication": "year",
                    "Publisher": "publisher",
                    "Image-URL-L": "image_url"
                },
                inplace=True
            )


            # ------------------------------------------------
            # RATINGS DATA
            # ------------------------------------------------

            ratings.rename(
                columns={
                    "User-ID": "user_id",
                    "Book-Rating": "rating"
                },
                inplace=True
            )


            # ------------------------------------------------
            # SELECT ACTIVE USERS
            # ------------------------------------------------

            x = ratings["user_id"].value_counts() > 200

            y = x[x].index

            ratings = ratings[
                ratings["user_id"].isin(y)
            ]


            # ------------------------------------------------
            # MERGE RATINGS WITH BOOKS
            # ------------------------------------------------

            ratings_with_books = ratings.merge(
                books,
                on="ISBN"
            )


            # ------------------------------------------------
            # COUNT RATINGS FOR EACH BOOK
            # ------------------------------------------------

            number_rating = (
                ratings_with_books
                .groupby("title")["rating"]
                .count()
                .reset_index()
            )

            number_rating.rename(
                columns={
                    "rating": "num_of_rating"
                },
                inplace=True
            )


            # ------------------------------------------------
            # ADD NUMBER OF RATINGS TO DATASET
            # ------------------------------------------------

            final_rating = ratings_with_books.merge(
                number_rating,
                on="title"
            )


            # ------------------------------------------------
            # KEEP BOOKS WITH AT LEAST 50 RATINGS
            # ------------------------------------------------

            final_rating = final_rating[
                final_rating["num_of_rating"] >= 50
            ]


            # ------------------------------------------------
            # REMOVE DUPLICATE USER-BOOK RATINGS
            # ------------------------------------------------

            final_rating.drop_duplicates(
                ["user_id", "title"],
                inplace=True
            )


            logging.info(
                f"Shape of final clean dataset: "
                f"{final_rating.shape}"
            )


            # ------------------------------------------------
            # SAVE CLEAN DATA
            # ------------------------------------------------

            clean_data_path = self.data_validation_config.clean_data_file

            os.makedirs(
                os.path.dirname(clean_data_path),
                exist_ok=True
            )

            final_rating.to_csv(
                clean_data_path,
                index=False
            )

            logging.info(
                f"Saved clean data to {clean_data_path}"
            )


        except Exception as e:

            raise AppException(e, sys) from e


    def initiate_data_validation(self):

        try:

            logging.info(
                f"{'=' * 20} "
                f"Data Validation log started. "
                f"{'=' * 20}"
            )

            self.preprocess_data()

            logging.info(
                f"{'=' * 20} "
                f"Data Validation log completed. "
                f"{'=' * 20}\n\n"
            )

        except Exception as e:

            raise AppException(e, sys) from e