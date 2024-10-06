import logging, os
from langgraph.managed import IsLastStep
from src.agent.toolkit.base import MastivTools
from src.ragpipeline.retrieval.retrieveknowledge import run_retriever
from src.utilities.helpers import load_yaml_file
from typing import Annotated, Literal, Sequence, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, HumanMessage
from src.utilities.messages import *
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# Set verbose mode to True by default
verbose = True
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep
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
        StreamConversation.updated_tools = MastivTools.call_tool()
        StreamConversation.prompt_template = self._load_prompt_template()
        StreamConversation.memory =  MemorySaver()

    def _load_prompt_template(self) -> str:
        """Load the instruction prompt template."""
        prompt_path = os.path.abspath("src/prompts/instruction.yaml")
        yaml_data = load_yaml_file(prompt_path)
        return yaml_data["INSTPROMPT"]
    

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
        try:
            prompt = ChatPromptTemplate.from_template(cls.prompt_template)
            rag_data = await run_retriever(message)
            if userData == {}:
                def modify_state_messages(state: AgentState):
                    return prompt.invoke({"messages": state["messages"],
                                                        "name":"",
                    "gender":"",
                    "timezone":"",
                    "current_location":"",
                    "rag_data":""})
            else:
                def modify_state_messages(state: AgentState):
                    return prompt.invoke({"messages": state["messages"],
                                                        "name":userData.get("name"),
                    "gender":userData.get("gender"),
                    "timezone":userData.get("timezone"),
                    "current_location":userData.get("current_location"),
                    "rag_data":rag_data})
            
            agent_executor = create_react_agent(cls.llm, cls.updated_tools, checkpointer=cls.memory,state_modifier=modify_state_messages)

            # Use the agent
            config = {"configurable": {"thread_id": "abc123"}}
            inputs = {"messages": [("user", message)]}
            for s in agent_executor.stream(inputs, stream_mode="values",config=config):
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                    yield f"data: {message}\n\n"
                else:
                    message.pretty_print()
        except Exception as e:
            logger.warning(
                "Error occurred while generating response: %s", str(e), exc_info=1
            )
            yield Mastiv_agent_executor_custom_response
 