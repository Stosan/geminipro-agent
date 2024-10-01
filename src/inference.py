import logging, os
from src.agent.planning_decision.agentchain import MastivAgent
from src.agent.planning_decision.parser import ReActSingleInputOutputParser
from src.agent.toolkit.base import MastivTools
from src.ragpipeline.retrieval.retrieveknowledge import run_retriever
from src.utilities.helpers import load_yaml_file
from typing import Literal
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from src.utilities.messages import *

logger = logging.getLogger(__name__)

# Set verbose mode to True by default
verbose = True

class StreamConversation:
    """
    A class to handle streaming conversation chains. It creates and stores memory for each conversation,
    and generates responses using the LLMs.
    """

    LLM = None

    def __init__(self, llm):
        """
        Initialize the StreamingConversation class.

        Args:
            llm: The language model for conversation generation.
        """
        self.llm = llm
        StreamConversation.LLM = llm
        self.memory = ConversationBufferMemory(return_messages=True)
        self.chat_history = self.memory.chat_memory.messages

    @classmethod
    def create_prompt(
        cls, message: str, name: str, gender: str, timezone: str, current_location: str
    ) -> (
        tuple[None, None, str]
        | tuple[
            None, None, Literal["something went wrong with retrieving vector store"]
        ]
        | tuple[str, AgentExecutor, ConversationBufferMemory, None]
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
            updated_tools = MastivTools.call_tool()
            prompt_path = os.path.abspath("src/prompts/instruction.yaml")
            INST_PROMPT = load_yaml_file(prompt_path)

            memory = ConversationBufferMemory(return_messages=True)

            agent = MastivAgent.load_llm_and_tools(
                cls.LLM,
                updated_tools,
                INST_PROMPT["INSTPROMPT"],
                ReActSingleInputOutputParser(),
                name,
                gender,
                timezone,
                current_location,
                rag_data=run_retriever(message)
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=updated_tools,
                memory=memory,
                max_iterations=8,
                handle_parsing_errors=True,
                verbose=verbose,
            )

            return message, agent_executor, memory, None
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
            message, agent_executor, memory, error_term = cls.create_prompt(message, "", "", "", "")
        else:
            message, agent_executor, memory, error_term = cls.create_prompt(
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

        try:
            input_data = {"input": message, "chat_history": cls.chat_history}

            _agent_response = await agent_executor.ainvoke(input_data)
            _agent_response_output = _agent_response.get("output")

            if isinstance(_agent_response_output, dict) and "output" in _agent_response_output:
                if "Thought: Do I need to use a tool?" in _agent_response_output["output"] or \
                   "Agent stopped due to iteration limit or time limit." in _agent_response_output["output"]:
                    yield Mastiv_agent_executor_custom_response
                else:
                    yield _agent_response_output["output"]
            elif isinstance(_agent_response_output, str):
                yield _agent_response_output
            else:
                for a in _agent_response_output:
                    yield a

            # Update memory with the conversation
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(_agent_response_output)

        except Exception as e:
            logger.warning(
                "Error occurred while generating response: %s", str(e), exc_info=1
            )
            yield Mastiv_agent_executor_custom_response