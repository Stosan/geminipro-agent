import logging, os
from src.agent.planning_decision.agentchain import MastivAgent
from src.agent.planning_decision.parser import ReActStructuredParser
from src.agent.toolkit.base import MastivTools
from src.ragpipeline.retrieval.retrieveknowledge import run_retriever
from src.utilities.helpers import load_yaml_file
from typing import Literal
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from src.utilities.messages import *

logger = logging.getLogger(__name__)

# Set verbose mode to True by default
verbose = True

class StreamConversation:
    """
    A class to handle streaming conversation chains. It creates and stores memory for each conversation,
    and generates responses using the LLMs.
    """

    def __init__(self, llm):
        """
        Initialize the StreamingConversation class.

        Args:
            llm: The language model for conversation generation.
        """
        StreamConversation.llm = llm
        StreamConversation.memory = ChatMessageHistory()
        StreamConversation.updated_tools = MastivTools.call_tool()
        StreamConversation.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the instruction prompt template."""
        prompt_path = os.path.abspath("src/prompts/instruction.yaml")
        yaml_data = load_yaml_file(prompt_path)
        return yaml_data["INSTPROMPT"]
    
    @classmethod
    async def create_prompt(
        cls, message: str, name: str, gender: str, timezone: str, current_location: str
    ) -> (
        tuple[None, None, str]
        | tuple[
            None, None, Literal["something went wrong with retrieving vector store"]
        ]
        | tuple[str, AgentExecutor, ChatMessageHistory, None]
        | tuple[Literal[""], None, None, str]
    ):
        """
        Create a prompt for the conversation.

        Args:
            message (str): The message to be added to the prompt.

        Returns:
            Tuple: A tuple containing message, agent_executor, memory, and an error term if any.
        """

        try:
            rag_data = await run_retriever(message)
            agent = MastivAgent.load_llm_and_tools(
                cls.llm,
                cls.updated_tools,
                cls.prompt_template,
                ReActStructuredParser(),
                name,
                gender,
                timezone,
                current_location,
                rag_data=rag_data
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=cls.updated_tools,
                max_iterations=8,
                handle_parsing_errors=True,
                verbose=verbose,
            )

            return message, agent_executor, cls.memory,None
        except Exception as e:
            logger.warning(
                "Error occurred while creating prompt: %s", str(e), exc_info=1
            )
            return "", None, None, str(e)

    @classmethod
    async def generate_response(cls, userData: dict, message: str):
        """
        Asynchronously generate a response for the conversation.

        Args:
            userData (dict): User data containing name, gender, timezone, and current_location.
            message (str): The user's message in the conversation.

        Yields:
            str: The generated response.
        """
        if userData == {}:
            message, agent_executor, memory, error_term = await cls.create_prompt(message, "", "", "", "")
        else:
            message, agent_executor, memory, error_term = await cls.create_prompt(
                message,
                userData.get("name"),
                userData.get("gender"),
                userData.get("timezone"),
                userData.get("current_location")
            )

        if error_term:
            yield error_term
            return

        if message == "":
            yield Mastiv_agent_executor_custom_response
            return

        if agent_executor is None:
            logger.warning("create_prompt must be called before generate_response", exc_info=1)
            return
        response_builder = ""
        try:
            input_data = {"input": message, "chat_history":memory.messages}
            async for _agent_response in agent_executor.astream(input_data):
                response_builder += _agent_response["output"]
                yield response_builder

        except Exception as e:
            logger.warning(
                "Error occurred while generating response: %s", str(e), exc_info=1
            )
            yield Mastiv_agent_executor_custom_response
        finally:
            # Update memory with the conversation
            memory.add_user_message(message)
            memory.add_ai_message(response_builder)