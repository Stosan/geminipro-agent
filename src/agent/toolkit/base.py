# Importing necessary libraries and modules
from typing import List
from src.agent.toolkit.bank import get_bank_account
from src.agent.toolkit.stockpricefinder import stock_price_finder
from src.agent.toolkit.google_search import create_google_tool
import logging

from src.utilities.Printer import printer
logger = logging.getLogger(__name__)

# Defining MastivTools class
class MastivTools:

    # Creating Google search tool
    google_tool = create_google_tool("Get Google Search",
                                     description="""Use this tool only when the task requires that you search online for the following reasons:
                                     1. The information is not provided to you.
                                     2. It is an infromation that is not recent or not available in your training data.
                                     3. It is a recurrent event and must likely have an updated version.""")

    # Creating Stock Price Finder tool
    stock_price_tool = stock_price_finder("Get Stock Prices",
                                     description="""This tool provides real-time stock price of companies. 
                                     Use this tool only when the task requires that you find the available stock price of any company, then check the user\'s bank account to see if they can afford it.
                                     Always rephrase the input word in the best search term to get the results.""")
    
    # Creating Bank Account tool
    bank_account_tool = get_bank_account("get bank account balance",
                                     description="""This tool retrieves the user\'s bank account balance for you to perform transctions and/or inquiries onbehalf of the user, especially, after a stock price enquiry.
                                     Input to this tool is already in the tool.""")
    

    # List of all tools
    toolkit = [google_tool, stock_price_tool, bank_account_tool]

    # Constructor for MastivTools class
    def __init__(self):
        # Displaying a message after loading preliminary tools
        printer(" ⚡️⚙️  preliminary tools::loaded","purple")

    # Method to call the appropriate tool based on the specified retriever
    @classmethod
    def call_tool(cls) -> List:
        """
        Calls the appropriate tools based on the specified retriever.

        Parameters:
        - retriever: Optional, the retriever to be used. If None, default tools are used.

        Returns:
        - List of tools to be executed.
        """
        try:
            tool_names=[]
            tools = cls.toolkit

            # Logging tool names
            for tl in tools:
                tool_names.append(tl.name)
            logger.info(f"""\nTools loaded Successfully!\n -------TOOL REPORT--------\ntool names: {str(tool_names)}\nnumber of tools: {len(tool_names)}\n---------------------\n""")

            return tools

     

        except Exception as e:
            # Log an error message if an exception occurs
            logger.warning(
                "Error occurred while creating tools: %s", str(e), exc_info=1
            )
