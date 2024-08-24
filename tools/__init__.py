from .annual_still_births_by_state import AnnualStillBirthsByStateAPIClient, AnnualStillBirthsByStateTool
from .annual_still_births import AnnualStillBirthsAPIClient, AnnualStillBirthsTool
from .annual_live_births_by_state import AnnualLiveBirthsByStateAPIClient, AnnualLiveBirthsByStateTool
from .annual_live_births import AnnualLiveBirthsAPIClient, AnnualLiveBirthsTool
from .daily_live_births import DailyLiveBirthsAPIClient, DailyLiveBirthsTool
from .car_registrations import CarRegistrationTool, CarRegistrationAPIClient

def get_tools():
    car_registration_tool = CarRegistrationTool(api_wrapper=CarRegistrationAPIClient())
    annual_live_births_tool = AnnualLiveBirthsTool(api_wrapper=AnnualLiveBirthsAPIClient())
    annual_live_births_by_state_tool = AnnualLiveBirthsByStateTool(api_wrapper=AnnualLiveBirthsByStateAPIClient())
    annual_still_births_tool = AnnualStillBirthsByStateTool(api_wrapper=AnnualStillBirthsByStateAPIClient())
    annual_still_births_by_state_tool = AnnualStillBirthsTool(api_wrapper=AnnualStillBirthsAPIClient())
    daily_live_births_tool = DailyLiveBirthsTool(api_wrapper=DailyLiveBirthsAPIClient())
    # append car registration tool to tools
    tools = [
        car_registration_tool,
        annual_live_births_tool,
        annual_live_births_by_state_tool,
        annual_still_births_tool,
        annual_still_births_by_state_tool,
        daily_live_births_tool
    ]

    return tools