import os

from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "./data")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "./data/raw")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "./data/processed")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
