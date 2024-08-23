from langchain.pydantic_v1 import BaseModel, Field, root_validator
from typing import Optional, Dict
from urllib.parse import urlencode, urljoin
from langchain.tools import BaseTool
import requests
import re
from langchain_core.callbacks import CallbackManagerForToolRun

class CarRegistrationAPIClient(BaseModel):
    base_url: str = "http://192.168.0.209"

    @root_validator(pre=True, allow_reuse=True)
    def validate_api_client(cls, values: Dict) -> Dict:
        print("validation done")
        return values
        

    def get_registration_info(self, params):
        base_url = f"{self.base_url}/registrations"
        if params:    
            query_string = urlencode(params)
            url = urljoin(base_url, f"?{query_string}")
        else:
            url = base_url
        response = requests.get(url)
        print("Get Car Registrations Response: ", response.status_code, response.json())
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

class CarRegistrationTool(BaseTool):
    name = "car_registration_data_lookup"
    description = """ A wrapper around car registration data lookup APIs.
    Useful to look up car registration information from 1st Jan, 2024 to 31st Jul, 2024.
    The api can look up car registration data by executing an HTTP GET endpoint. 
    Optionally the GET endpoint can accept query parameters to filter the results.
    The acceptable query parameters are:
    - dateFrom: The start date of the registration period. Date format: YYYY-MM-DD
    - dateTo: The end date of the registration period. Date format: YYYY-MM-DD
    - type: The type of the car. (e.g. jip, pick_up, motokar, etc.)
    - make: The make of the car.
    - model: The model of the car.
    - color: The color of the car.
    - fuel: The fuel type of the car.
    - state: The state where the car is registered.
    - count: set to true, if you just need the count instead of actual data.
    """
    api_wrapper: CarRegistrationAPIClient


    def _run(
            self, 
            input: Optional[str]=None,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
        print("Entering CarRegistrationTool run function")
        params = {}
        if input:
            matches = re.findall(r'(\w+)=([\w\-]+)', input)
            params = dict(matches)
        print("Invoking car registration data look up tool with params: ", params)
        return self.api_wrapper.get_registration_info(params)

    def _arun(self, params):
        raise NotImplementedError("Asynchronous operation is not supported for this tool.")

