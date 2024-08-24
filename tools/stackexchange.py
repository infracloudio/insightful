from langchain_community.utilities import StackExchangeAPIWrapper
from langchain_community.tools.stackexchange.tool import StackExchangeTool


def tool():
    stackexchange_wrapper = StackExchangeAPIWrapper(max_results=3)
    return StackExchangeTool(api_wrapper=stackexchange_wrapper)
