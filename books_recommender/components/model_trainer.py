import os
import sys
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from books_recommender.log_details.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException


class ModelTrainer:

    def __init__(self, app_config=AppConfiguration()):
        try:
            self.model_trainer_config = app_config.get_model_trainer_config()
            self.data_transformation_config = app_config.get_data_transformation_config()
        except Exception as e:
            raise AppException(e, sys) from e


    def train(self):
        try:
            # Load transformed pivot table
            book_pivot = pickle.load(
                open(self.data_transformation_config.book_pivot_file, "rb")
            )

            book_sparse = csr_matrix(book_pivot)

            # Train KNN model
            model = NearestNeighbors(algorithm="brute")
            model.fit(book_sparse)

            # Create model directory
            os.makedirs(
                self.model_trainer_config.model_dir,
                exist_ok=True
            )

            # Save trained model
            pickle.dump(
                model,
                open(self.model_trainer_config.trained_model_path, "wb")
            )

            logging.info(
                f"Saved trained model to {self.model_trainer_config.trained_model_path}"
            )

        except Exception as e:
            raise AppException(e, sys) from e


    def initiate_model_trainer(self):
        try:
            logging.info(f"{'='*20}Model Trainer started.{'='*20}")

            self.train()

            logging.info(f"{'='*20}Model Trainer completed.{'='*20}\n\n")

        except Exception as e:
            raise AppException(e, sys) from e