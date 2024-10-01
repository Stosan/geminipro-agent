import pytest
from datetime import datetime

from src.utilities.helpers import check_final_answer_exist, get_day_date_month_year_time, load_yaml_file


def test_load_yaml_file(mocker):
    # Mock the open function and yaml.safe_load
    mock_yaml_content = "key: value\nlist:\n  - item1\n  - item2"
    expected_result = {"key": "value", "list": ["item1", "item2"]}
    
    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_yaml_content))
    mocker.patch("yaml.safe_load", return_value=expected_result)
    
    result = load_yaml_file("dummy_path.yaml")
    assert result == expected_result

def test_get_day_date_month_year_time(mocker):
    # Set a fixed datetime for testing
    mock_datetime = mocker.patch('your_module.dts')
    mock_datetime.now.return_value = datetime(2023, 4, 15, 10, 30, 45)
    
    result = get_day_date_month_year_time()
    
    expected = ('04-15-2023', 'Saturday', 15, 4, 2023, 10, 30, 45)
    assert result == expected

@pytest.mark.parametrize("test_input,expected", [
    ("This is the final answer", True),
    ("FINAL ANSWER: correct", True),
    ("The answer_final is here", True),
    ("final_answer found", True),
    ("ANSWER FINAL: done", True),
    ("This is the finale", False),
    ("The answer is finalizing", False),
    ("Finally answered", False),
    ("final result", False),
    ("answer completed", False)
])
def test_check_final_answer_exist(test_input, expected):
    assert check_final_answer_exist(test_input) == expected