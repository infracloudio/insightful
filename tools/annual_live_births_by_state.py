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

class AnnualLiveBirthsByStateAPIClient(BaseModel):
    base_url: str = os.getenv("TOOLS_BASE_URL", default="http://localhost:5000")

    @root_validator(pre=True, allow_reuse=True)
    def validate_api_client(cls, values: Dict) -> Dict:
        logging.info("validation done AnnualLiveBirthsByStateAPIClient")
        return values
        

    def get_annual_live_births_by_state_info(self, params):
        base_url = f"{self.base_url}/annual_live_births_by_state"
        if params:    
            query_string = urlencode(params)
            url = urljoin(base_url, f"?{query_string}")
        else:
            url = base_url
        response = requests.get(url)
        logging.info(f"Annual Live Births By State Response code: {response.status_code}, response json: {response.json()}")
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

class AnnualLiveBirthsByStateTool(BaseTool):
    name = "annual_live_births_by_state_data_lookup"
    description = """ A wrapper around annual live births, i.e. children born alive, by state data lookup APIs, for the country of Malaysia.
    Useful to look up annual children born alive per state information from 1st Jan, 2000 to 1st Jan, 2022.
    The data contains just one row or less,  per year per state, with the number, i.e. absolute, abs in short, of children born alive in that year.
    The api can look up annual live births data per state by executing an HTTP GET endpoint. 
    Optionally the GET endpoint can accept query parameters to filter the results.
    
    You MUST use this tool ONLY to fetch state-wise live births data. For fetching data for the entire country, use the annual_live_births tool.
    
    Args:
        A JSON string with the following fields:
        - state: The state where the children are born. Skip this field to get data for all states.
        - dateFrom: The start date of the registration period. Date format: YYYY-MM-DD
        - dateTo: The end date of the registration period. Date format: YYYY-MM-DD
        - abs: The absolute number of children born alive.
        - absMin: The minimum absolute number of children born alive.
        - absMax: The maximum absolute number of children born alive.
        - rate: The rate of children born alive.
        - rateMin: The minimum rate of children born alive.
        - rateMax: The maximum rate of children born alive.
    
    Returns:
        A JSON string with the following fields:
        - state: The state where the children are born.
        - date: The date of birth record in the format YYYY-MM-DD.
        - abs: The absolute number of children born alive.
        - rate: The rate of children born alive.
    
    """
    api_wrapper: AnnualLiveBirthsByStateAPIClient


    def _run(
            self, 
            input: Optional[str]=None,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
        logging.info(f"\nEntering AnnualLiveBirthsByStateTool run function\nOriginal input: {input}\n")
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
            
        logging.info(f"\nInvoking annual live births by state data look up tool with params: {params}")
        return self.api_wrapper.get_annual_live_births_by_state_info(params)

    def _arun(self, params):
        raise NotImplementedError("Asynchronous operation is not supported for this tool.")

