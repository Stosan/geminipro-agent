import re
from typing import Union, Any
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser
from langchain_core.exceptions import OutputParserException
import logging

logger = logging.getLogger(__name__)

FORMAT_INSTRUCTIONS = """Your response should be in one of two formats:

1. If you have a final answer, respond with:
{
    "mastiveairesponse": "Your final answer here"
}

2. If you need to take an action, respond with:
{
    "Thought": "Your thought process here",
    "Action": "The action to take",
    "Input": "The input for the action",
    "Observation": "Your observation about the action and its result"
}
"""

def check_and_process_string(text):
    pattern1 = r'\{\s*"mastiveairesponse":\s*"([^"]*)"\s*\}'
    pattern2 = r'\{\s*"Thought":\s*(.+?)\s*"Action":\s*(.+?)\s*"Input":\s*(.+?)\s*"Observation":\s*(.+?)\s*\}'

    match1 = re.search(pattern1, text, re.DOTALL)
    if match1:
        return match1.group(1)
    else:
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            # Extract values and convert to JSON
            json_structure = {
                "Thought": match2.group(1).strip(),
                "Action": match2.group(2).strip(),
                "Input": match2.group(3).strip(),
                "Observation": match2.group(4).strip()
            }
            return json_structure
        else:
            return False

class ReActStructuredParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        res = check_and_process_string(text)
        
        if isinstance(res, str):
            print(f"Final answer: {res}")
            return AgentFinish(
                {"output": res}, res
            )
        elif isinstance(res, dict):
            return AgentAction(res["Action"], res["Input"], text)
        else:
            logger.error(
                "Error occurred while parsing output: \nCould not parse LLM output: %s", text, exc_info=1)
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation="Unknown type returned from check_and_process_string",
                llm_output=text,
                send_to_llm=True,
            )






















# import json
# import re,logging
# from typing import Union

# from langchain_core.agents import AgentAction, AgentFinish
# from langchain_core.exceptions import OutputParserException

# from langchain.agents.agent import AgentOutputParser
# from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
# logger = logging.getLogger(__name__)


# FINAL_ANSWER_ACTION = "Final Answer:"
# MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
#     "Invalid Format: Missing 'Action:' after 'Thought:"
# )
# MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
#     "Invalid Format: Missing 'Input:' after 'Action:'"
# )
# FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
#     "Parsing LLM output produced both a final answer and a parse-able action:"
# )

# def check_and_process_string(text):
#     pattern1 = r'\{\s*"mastiveairesponse":\s*"([^"]*)"\s*\}'
#     pattern2 = r'\{\s*"Thought":\s*(.+?)\s*"Action":\s*(.+?)\s*"Input":\s*(.+?)\s*"Observation":\s*(.+?)\s*\}'

#     match1 = re.search(pattern1, text, re.DOTALL)
#     if match1:
#        return match1.group(1)
#     else:
#         match2 = re.search(pattern2, text, re.DOTALL)
#         if match2:
#             # Extract values and convert to JSON
#             json_structure = {
#                 "Thought": match2.group(1).strip(),
#                 "Action": match2.group(2).strip(),
#                 "Input": match2.group(3).strip(),
#                 "Observation": match2.group(4).strip()
#             }
#             return json_structure
#         else:
#             return False
    

# class ReActSingleInputOutputParser(AgentOutputParser):

#     def get_format_instructions(self) -> str:
#         return FORMAT_INSTRUCTIONS

#     def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
#         print(type(text))
#         res = check_and_process_string(text)
#         print(res)
#         match type(res):
#             case type(str):
#                 print(res)
#                 return AgentFinish(
#                     {"output": res}, res
#                 )
#             case type(dict):
#                 return AgentAction(res["Action"], res["Input"], text)
#             case Any:
#                 logger.error(
#                     "Error occurred while parsing output: \nCould not parse LLM output: %s", text, exc_info=1)
#                 raise OutputParserException(
#                     f"Could not parse LLM output: `{text}`",
#                     observation="Unknown type returned from check_and_process_string",
#                     llm_output=text,
#                     send_to_llm=True,
#                 )
        
#         # includes_answer = FINAL_ANSWER_ACTION in text
#         # regex = (
#         #     r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Input\s*\d*\s*:[\s]*(.*)"
#         # )
        
#         # action_match = re.search(regex, text, re.DOTALL)
#         # if action_match:
#         #     if includes_answer:
#         #         logger.error("Error occurred while parsing output: \nFINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE: %s", text, exc_info=1)
#         #         if "Action:" in text and "Input:" in text:
#         #             prunned_text = "Action:"+text.split("Action:")[1]
#         #             action_match = re.search(regex, prunned_text, re.DOTALL)
#         #     action = action_match.group(1).strip()
#         #     tool_input = action_match.group(2).strip()
#         #     return AgentAction(action, tool_input, text)
        
#         # elif 'Action:' not in text  and 'Input:' not in text:
#         #     return AgentFinish(
#         #         {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
#         #     )
        
#         # elif includes_answer:
#         #     return AgentFinish(
#         #         {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
#         #     )
#         # if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
#         #     logger.error(
#         #     "Error occurred while parsing output: \nCould not parse LLM output: %s", "MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE", exc_info=1)
#         #     raise OutputParserException(
#         #         f"Could not parse LLM output: `{text}`",
#         #         observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
#         #         llm_output=text,
#         #         send_to_llm=True,
#         #     )
#         # elif not re.search(r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL):
#         #     raise OutputParserException(
#         #         f"Could not parse LLM output: `{text}`",
#         #         observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
#         #         llm_output=text,
#         #         send_to_llm=True,
#         #     )
                            
#         # else:
#         #     logger.error(
#         #     "Error occurred while parsing output: \nCould not parse LLM output: %s", text, exc_info=1)

#     @property
#     def _type(self) -> str:
#         return "react-single-input"


