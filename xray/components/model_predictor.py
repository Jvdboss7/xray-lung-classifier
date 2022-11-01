import logging
import sys
from typing import Dict
#from pandas import DataFrame
#import pandas as pd
from xray.constants import *
from xray.configuration.s3_operations import S3Operation
from xray.exception import XrayException

# initializing logger
logger = logging.getLogger(__name__)

class XrayData:
    def __init__(self,images,):
        self.images = images

    def get_data(self) -> Dict:

        """
        Method Name :   get_data
        Description :   This method gets data. 
        
        Output      :    Input data in dictionary
        """
        logger.info("Entered get_data method of SensorData class")
        try:
            # Saving the features as dictionary
            input_data = {
                "images": [self.images],
            }

            logger.info("Exited get_data method of XrayData class")
            return input_data

        except Exception as e:
            raise XrayException(e, sys)