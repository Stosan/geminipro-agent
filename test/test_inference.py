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
        assert isinstance(agent_executor, AgentExecutor)
        assert isinstance(chat_history, MagicMock)  # ChatMessageHistory is now mocked
        assert error is None

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

@pytest.mark.asyncio
async def test_generate_response_success(stream_conversation):
    user_data = {
        "name": "John",
        "gender": "Male",
        "timezone": "UTC",
        "current_location": "New York"
    }
    
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt, \
         patch.object(AgentExecutor, 'astream') as mock_astream:
        
        mock_create_prompt.return_value = ("Hello", MagicMock(), MagicMock(), None)
        mock_astream.return_value = AsyncMock()
        mock_astream.return_value.__aiter__.return_value = [{"output": "Test response"}]

        response = [chunk async for chunk in StreamConversation.generate_response(user_data, "Hello")]

        assert response == ["Test response"]
        mock_create_prompt.assert_called_once_with("Hello", "John", "Male", "UTC", "New York")

@pytest.mark.asyncio
async def test_generate_response_empty_user_data(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt, \
         patch.object(AgentExecutor, 'astream') as mock_astream:
        
        mock_create_prompt.return_value = ("Hello", MagicMock(), MagicMock(), None)
        mock_astream.return_value = AsyncMock()
        mock_astream.return_value.__aiter__.return_value = [{"output": "Test response"}]

        response = [chunk async for chunk in StreamConversation.generate_response({}, "Hello")]

        assert response == ["Test response"]
        mock_create_prompt.assert_called_once_with("Hello", "", "", "", "")

@pytest.mark.asyncio
async def test_generate_response_error_in_create_prompt(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt:
        mock_create_prompt.return_value = ("", None, None, "Test error")

        response = [chunk async for chunk in StreamConversation.generate_response({}, "Hello")]

        assert response == ["Test error"]

@pytest.mark.asyncio
async def test_generate_response_empty_message(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt:
        mock_create_prompt.return_value = ("", MagicMock(), MagicMock(), None)

        response = [chunk async for chunk in StreamConversation.generate_response({}, "")]

        assert response == ["I apologize, but I encountered an issue while processing your request. Could you please try again or rephrase your question?"]

@pytest.mark.asyncio
async def test_generate_response_agent_executor_none(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt:
        mock_create_prompt.return_value = ("Hello", None, MagicMock(), None)

        response = [chunk async for chunk in StreamConversation.generate_response({}, "Hello")]

        assert response == ["I apologize, but I encountered an issue while processing your request. Could you please try again or rephrase your question?"]

@pytest.mark.asyncio
async def test_generate_response_exception_in_astream(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt, \
         patch.object(AgentExecutor, 'astream', side_effect=Exception("Test error")):
        
        mock_create_prompt.return_value = ("Hello", MagicMock(), MagicMock(), None)

        response = [chunk async for chunk in StreamConversation.generate_response({}, "Hello")]

        assert response == ["I apologize, but I encountered an issue while processing your request. Could you please try again or rephrase your question?"]