from src.exception import CustomException
import sys

try:
    pass
except Exception as e:
    raise CustomException(e, sys)