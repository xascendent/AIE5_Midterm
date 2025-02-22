from typing import List, Dict
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults



from typing import Annotated, List, Tuple, Union
from langchain_core.tools import tool


tavily_tool = TavilySearchResults(max_results=5)

