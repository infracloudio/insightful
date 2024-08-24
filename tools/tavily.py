from langchain_community.tools.tavily_search import TavilySearchResults


def tool():
    return TavilySearchResults(max_results=10, handle_tool_error=True)
