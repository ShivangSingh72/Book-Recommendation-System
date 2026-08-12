# constant are things that almost never change and are used throughout the project.
# Constant files - folders roots

import os

ROOT_DIR = os.getcwd()

CONFIG_FOLDER_NAME = "config"
CONFIG_FILE_NAME = "config.yaml"

CONFIG_FILE_PATH = os.path.join(
    ROOT_DIR,
    CONFIG_FOLDER_NAME,
    CONFIG_FILE_NAME
)