import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agent.base.planning_decision import MastivAgent
from src.agent.toolkit.base import MastivTools
from langchain.agents import AgentExecutor

# Import the StreamConversation class (assuming it's in a file named stream_conversation.py)
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
         patch('src.agent.base.agenthead.MastivAgent.load_llm_and_tools') as mock_load_llm_and_tools:
        
        mock_call_tool.return_value = []
        mock_load_yaml.return_value = {"INSTPROMPT": "Test prompt"}
        mock_load_llm_and_tools.return_value = MagicMock()

        message, agent_executor, chat_history, error = StreamConversation.create_prompt(
            "Hello", "John", "Male", "UTC", "New York"
        )

        assert message == "Hello"
        assert isinstance(agent_executor, AgentExecutor)
        assert chat_history == []
        assert error is None

@pytest.mark.asyncio
async def test_create_prompt_failure(stream_conversation):
    with patch('src.agent.toolkit.base.MastivTools.call_tool', side_effect=Exception("Test error")):
        message, agent_executor, chat_history, error = StreamConversation.create_prompt(
            "Hello", "John", "Male", "UTC", "New York"
        )

        assert message == ""
        assert agent_executor is None
        assert chat_history is None
        assert error == "Test error"

@pytest.mark.asyncio
async def test_generate_response_success(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt, \
         patch.object(AgentExecutor, 'ainvoke') as mock_ainvoke:
        
        mock_create_prompt.return_value = ("Hello", MagicMock(), [], None)
        mock_ainvoke.return_value = {"output": "Test response"}

        response = await StreamConversation.generate_response({}, "Hello")

        assert response == "Test response"

@pytest.mark.asyncio
async def test_generate_response_with_user_data(stream_conversation):
    user_data = {
        "name": "John",
        "gender": "Male",
        "timezone": "UTC",
        "current_location": "New York"
    }
    
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt, \
         patch.object(AgentExecutor, 'ainvoke') as mock_ainvoke:
        
        mock_create_prompt.return_value = ("Hello", MagicMock(), [], None)
        mock_ainvoke.return_value = {"output": "Test response with user data"}

        response = await StreamConversation.generate_response(user_data, "Hello")

        assert response == "Test response with user data"
        mock_create_prompt.assert_called_once_with("Hello", "John", "Male", "UTC", "New York")

@pytest.mark.asyncio
async def test_generate_response_error_in_create_prompt(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt:
        mock_create_prompt.return_value = ("", None, None, "Test error")

        response = await StreamConversation.generate_response({}, "Hello")

        assert response == "Test error"

@pytest.mark.asyncio
async def test_generate_response_empty_message(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt:
        mock_create_prompt.return_value = ("", MagicMock(), [], None)

        response = await StreamConversation.generate_response({}, "")

        assert response == "I apologize, but I encountered an issue while processing your request. Could you please try again or rephrase your question?"

@pytest.mark.asyncio
async def test_generate_response_agent_executor_none(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt:
        mock_create_prompt.return_value = ("Hello", None, [], None)

        response = await StreamConversation.generate_response({}, "Hello")

        assert response == "I apologize, but I encountered an issue while processing your request. Could you please try again or rephrase your question?"

@pytest.mark.asyncio
async def test_generate_response_exception_in_ainvoke(stream_conversation):
    with patch.object(StreamConversation, 'create_prompt') as mock_create_prompt, \
         patch.object(AgentExecutor, 'ainvoke', side_effect=Exception("Test error")):
        
        mock_create_prompt.return_value = ("Hello", MagicMock(), [], None)

        response = await StreamConversation.generate_response({}, "Hello")

        assert response == "I apologize, but I encountered an issue while processing your request. Could you please try again or rephrase your question?"