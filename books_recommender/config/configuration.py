# Bridge bw config.yaml and rest of ml project

# config.yaml                         
#     ↓
# read_yaml_file()
#     ↓
# AppConfiguration
#     ↓
# Config objects
#     ↓
# ML components

# | File               | Job                                       |
# | ------------------ | ----------------------------------------- |
# | `config.yaml`      | Stores the settings                       |
# | `config_entity.py` | Defines the structure of those settings   |
# | `configuration.py` | Reads YAML and creates the config objects |


import os
import sys

from books_recommender.constant import CONFIG_FILE_PATH
from books_recommender.exception.exception_handler import AppException
from books_recommender.log_details.log import logging
from books_recommender.entity.config_entity import (
    ArtifactsConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelRecommendationConfig,
    ModelTrainerConfig
)
from books_recommender.utils.utils import read_yaml_file


class AppConfiguration:
    #read yaml file - (self.configs_info is basically a Python dictionary containing everything from your YAML)
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH):
        try:
            self.configs_info = read_yaml_file(
                file_path=config_file_path
            )

        except Exception as e:
            raise AppException(e, sys) from e


    def get_artifacts_config(self) -> ArtifactsConfig:

        try:
            artifacts_config = self.configs_info["artifacts_config"]

            artifacts_dir = artifacts_config["artifacts_dir"]

            response = ArtifactsConfig(
                artifacts_dir=artifacts_dir
            )

            logging.info(f"Artifacts Config: {response}")

            return response

        except Exception as e:
            raise AppException(e, sys) from e


    def get_data_ingestion_config(self) -> DataIngestionConfig:

        try:
            data_ingestion_config = self.configs_info[
                "data_ingestion_config"
            ]

            data_dir = data_ingestion_config["data_dir"]

            books_csv_file = os.path.join(
                data_dir,
                data_ingestion_config["books_csv_file"]
            )

            users_csv_file = os.path.join(
                data_dir,
                data_ingestion_config["users_csv_file"]
            )

            ratings_csv_file = os.path.join(
                data_dir,
                data_ingestion_config["ratings_csv_file"]
            )

            response = DataIngestionConfig(
                data_dir=data_dir,
                books_csv_file=books_csv_file,
                users_csv_file=users_csv_file,
                ratings_csv_file=ratings_csv_file
            )

            logging.info(f"Data Ingestion Config: {response}")

            return response

        except Exception as e:
            raise AppException(e, sys) from e


    def get_data_validation_config(self) -> DataValidationConfig:

        try:

            artifacts_dir = self.configs_info["artifacts_config"]["artifacts_dir"]

            data_ingestion_config = self.configs_info["data_ingestion_config"]

            data_validation_config = self.configs_info["data_validation_config"]

            books_csv_file = os.path.join(
            data_ingestion_config["data_dir"],
            data_ingestion_config["books_csv_file"]
        )

            users_csv_file = os.path.join(
            data_ingestion_config["data_dir"],
            data_ingestion_config["users_csv_file"]
        )

            ratings_csv_file = os.path.join(
            data_ingestion_config["data_dir"],
            data_ingestion_config["ratings_csv_file"]
        )

            clean_data_file = os.path.join(
                artifacts_dir,
                data_validation_config["clean_data_file"]
            )

            response = DataValidationConfig(
            books_csv_file=books_csv_file,
            users_csv_file=users_csv_file,
            ratings_csv_file=ratings_csv_file,
            clean_data_file=clean_data_file
        )

            logging.info(f"Data Validation Config: {response}")

            return response

        except Exception as e:
            raise AppException(e, sys) from e


    def get_data_transformation_config(self) -> DataTransformationConfig:

        try:
            artifacts_dir = self.configs_info["artifacts_config"]["artifacts_dir"]
            data_validation_config = self.configs_info["data_validation_config"]
            data_transformation_config = self.configs_info["data_transformation_config"]

            clean_data_file = os.path.join(
                artifacts_dir,
                data_validation_config["clean_data_file"]
        )

            final_rating_file = os.path.join(
                artifacts_dir,
                data_transformation_config["final_rating_file"]
        )

            book_pivot_file = os.path.join(
                artifacts_dir,
                data_transformation_config["book_pivot_file"]
        )

            book_names_file = os.path.join(
                artifacts_dir,
                data_transformation_config["book_names_file"]
        )

            response = DataTransformationConfig(
                clean_data_file=clean_data_file,
                final_rating_file=final_rating_file,
                book_pivot_file=book_pivot_file,
                book_names_file=book_names_file
        )

            logging.info(f"Data Transformation Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:

        try:
            model_trainer_config = self.configs_info[
                "model_trainer_config"
            ]

            model_dir = model_trainer_config["model_dir"]

            model_name = model_trainer_config["model_name"]

            trained_model_path = os.path.join(
                model_dir,
                model_name
            )

            response = ModelTrainerConfig(
                model_dir=model_dir,
                model_name=model_name,
                trained_model_path=trained_model_path
            )

            logging.info(f"Model Trainer Config: {response}")

            return response

        except Exception as e:
            raise AppException(e, sys) from e

    def get_recommendation_config(self) -> ModelRecommendationConfig:
        try:
            artifacts_dir = self.configs_info["artifacts_config"]["artifacts_dir"]
            data_transformation = self.configs_info["data_transformation_config"]
            model_trainer = self.configs_info["model_trainer_config"]

            response = ModelRecommendationConfig(
                book_names_file=os.path.join(
                    artifacts_dir,
                    data_transformation["book_names_file"]
            ),
                book_pivot_file=os.path.join(
                    artifacts_dir,
                    data_transformation["book_pivot_file"]
            ),
                final_rating_file=os.path.join(
                    artifacts_dir,
                    data_transformation["final_rating_file"]
            ),
                trained_model_path=os.path.join(
                    model_trainer["model_dir"],
                    model_trainer["model_name"]
            )
        )

            logging.info(f"Recommendation Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e