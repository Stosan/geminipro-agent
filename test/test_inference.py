import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agent.planning_decision.agentchain import MastivAgent
from src.agent.toolkit.base import MastivTools
from langchain.agents import AgentExecutor
from src.inference import StreamConversation

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def stream_conversation(mock_llm):
    return StreamConversation(mock_llm)

@pytest.mark.asyncio
async def test_create_prompt_success(stream_conversation):
    with patch('src.agent.toolkit.base.MastivTools.call_tool') as mock_call_tool, \
         patch('src.utilities.helpers.load_yaml_file') as mock_load_yaml, \
         patch('src.agent.planning_decision.agentchain.MastivAgent.load_llm_and_tools') as mock_load_llm_and_tools, \
         patch('src.ragpipeline.retrieval.retrieveknowledge.run_retriever') as mock_run_retriever:
        
        mock_call_tool.return_value = []
        mock_load_yaml.return_value = {"INSTPROMPT": "Test prompt"}
        mock_load_llm_and_tools.return_value = MagicMock()
        mock_run_retriever.return_value = "Mock RAG data"

        message, agent_executor, chat_history, error = await StreamConversation.create_prompt(
            "Hello", "John", "Male", "UTC", "New York"
        )

        assert message == "Hello"
        # assert isinstance(agent_executor, AgentExecutor)
        # assert isinstance(chat_history, MagicMock)  # ChatMessageHistory is now mocked
        # assert error is None

@pytest.mark.asyncio
async def test_create_prompt_failure(stream_conversation):
    with patch('src.ragpipeline.retrieval.retrieveknowledge.run_retriever', side_effect=Exception("Test error")):
        message, agent_executor, chat_history, error = await StreamConversation.create_prompt(
            "Hello", "John", "Male", "UTC", "New York"
        )

        assert message == ""
        assert agent_executor is None
        assert chat_history is None
        assert error == "Test error"
