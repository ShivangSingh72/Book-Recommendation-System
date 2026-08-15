#YAML
# ├── data_dir
# ├── books_file
# ├── users_file
# └── ratings_file

# Python reads it as a dictionary.
# config_entity.py converts that dictionary into a proper object

from collections import namedtuple


ArtifactsConfig = namedtuple(
    "ArtifactsConfig",
    [
        "artifacts_dir"
    ]
)


DataIngestionConfig = namedtuple(
    "DataIngestionConfig",
    [
        "data_dir",
        "books_csv_file",
        "users_csv_file",
        "ratings_csv_file"
    ]
)

DataValidationConfig = namedtuple(
    "DataValidationConfig",
    [
        "books_csv_file",
        "users_csv_file",
        "ratings_csv_file",
        "clean_data_file"
    ]
)

DataTransformationConfig = namedtuple(
    "DataTransformationConfig",
    [
        "clean_data_file",
        "final_rating_file",
        "book_pivot_file",
        "book_names_file"
    ]
)


ModelTrainerConfig = namedtuple(
    "ModelTrainerConfig",
    [
        "model_dir",
        "model_name",
        "trained_model_path"
    ]
)

ModelRecommendationConfig = namedtuple(
    "ModelRecommendationConfig",
    [
        "book_names_file",
        "book_pivot_file",
        "final_rating_file",
        "trained_model_path"
    ]
)