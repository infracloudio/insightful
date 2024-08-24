from langchain.pydantic_v1 import BaseModel, Field, root_validator
from typing import Optional, Dict
from urllib.parse import urlencode, urljoin
from langchain.tools import BaseTool
import logging
import os
import json
from tools.utils import cleanupInputAndGetDictFromStr
import requests
import re
from langchain_core.callbacks import CallbackManagerForToolRun

logger = logging.getLogger(__name__)

class AnnualStillBirthsAPIClient(BaseModel):
    base_url: str = os.getenv("TOOLS_BASE_URL", default="http://localhost:5000")

    @root_validator(pre=True, allow_reuse=True)
    def validate_api_client(cls, values: Dict) -> Dict:
        logging.info("validation done AnnualStillBirthsAPIClient")
        return values
        

    def get_annual_still_births_info(self, params):
        base_url = f"{self.base_url}/annual_still_births"
        if params:    
            query_string = urlencode(params)
            url = urljoin(base_url, f"?{query_string}")
        else:
            url = base_url
        response = requests.get(url)
        logging.info(f"Annual Still Births Response code: {response.status_code}, response json: {response.json()}")
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

class AnnualStillBirthsTool(BaseTool):
    name = "annual_still_births_data_lookup"
    description = """ A wrapper around api to lookup annual still births in the whole country of Malaysia.
    Useful to look up annual stillborn children information from 1st Jan, 2000 to 1st Jan, 2022.
    The data contains just one row per year, with the number, i.e. absolute, abs in short, of stillborn children in that year.
    The api can look up annual still births data by executing an HTTP GET endpoint. 
    Optionally the GET endpoint can accept query parameters to filter the results.
    
    You MUST use this tool ONLY to fetch country-wise live births data. To fetch state-wise data, use the annual_still_births_by_state tool.
    
    Args:
        A JSON string with the following fields:
        - dateFrom: The start date of the registration period. Date format: YYYY-MM-DD
        - dateTo: The end date of the registration period. Date format: YYYY-MM-DD
        - abs: The absolute number of stillborn children.
        - absMin: The minimum absolute number of stillborn children.
        - absMax: The maximum absolute number of stillborn children.
        - rate: The rate of stillborn children.
        - rateMin: The minimum rate of stillborn children.
        - rateMax: The maximum rate of stillborn children.
        - count: set to true, if you just need the count instead of actual data.
    
    Returns:
        A JSON string with the following fields:
        - date: The date of birth record in the format YYYY-MM-DD.
        - abs: The absolute number of stillborn children.
        - rate: The rate of stillborn children.
    """
    api_wrapper: AnnualStillBirthsAPIClient


    def _run(
            self, 
            input: Optional[str]=None,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
        logging.info(f"\nEntering AnnualStillBirthsTool run function\nOriginal input: {input}\n")
        params = {}
        if input:
            matches = re.findall(r'(\w+)=([\w\-]+)', input)
            params = dict(matches)
            matches = re.findall(r'(\w+): ([\w\d\-]+)', input)
            params.update(dict(matches))
            if not params:
                params = cleanupInputAndGetDictFromStr(input)
            
            if str(params.get("count")) != "None":
                params.pop("count")
            
        logging.info(f"Invoking annual still births data look up tool with params: {params}")
        return self.api_wrapper.get_annual_still_births_info(params)

    def _arun(self, params):
        raise NotImplementedError("Asynchronous operation is not supported for this tool.")
