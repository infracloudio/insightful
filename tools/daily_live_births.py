from langchain.pydantic_v1 import BaseModel, Field, root_validator
from typing import Optional, Dict
from urllib.parse import urlencode, urljoin
from langchain.tools import BaseTool
import requests
import re
from langchain_core.callbacks import CallbackManagerForToolRun

class DailyLiveBirthsAPIClient(BaseModel):
    base_url: str = "http://192.168.0.209"

    @root_validator(pre=True, allow_reuse=True)
    def validate_api_client(cls, values: Dict) -> Dict:
        print("validation done")
        return values
        

    def get_daily_live_births_info(self, params):
        base_url = f"{self.base_url}/daily_live_births"
        if params:    
            query_string = urlencode(params)
            url = urljoin(base_url, f"?{query_string}")
        else:
            url = base_url
        response = requests.get(url)
        print("Daily Live Births Response: ", response.status_code, response.json())
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

class DailyLiveBirthsTool(BaseTool):
    name = "daily_live_births_data_lookup"
    description = """ A wrapper around daily live births, i.e. children born alive, data lookup APIs.
    Useful to look up daily children born alive information from 1st Jan, 1920 to 31st Jul, 2023.
    The data contains just one row per day, with the number, i.e. births, of children born alive in that day.
    The api can look up daily live births data by executing an HTTP GET endpoint. 
    Optionally the GET endpoint can accept query parameters to filter the results.
    The acceptable query parameters are:
    - dateFrom: The start date of the registration period. Date format: YYYY-MM-DD
    - dateTo: The end date of the registration period. Date format: YYYY-MM-DD
    - births: The number of children born alive on a day.
    - birthsMin: The minimum number of children born alive on a day.
    - birthsMax: The maximum number of children born alive on a day.
    - count: set to true, if you just need the count instead of actual data.
    """
    api_wrapper: DailyLiveBirthsAPIClient


    def _run(
            self, 
            input: Optional[str]=None,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
        print("Entering DailyLiveBirthsTool run function")
        params = {}
        if input:
            matches = re.findall(r'(\w+)=([\w\-]+)', input)
            params = dict(matches)
        print("Invoking daily live births data look up tool with params: ", params)
        return self.api_wrapper.get_daily_live_births_info(params)

    def _arun(self, params):
        raise NotImplementedError("Asynchronous operation is not supported for this tool.")

