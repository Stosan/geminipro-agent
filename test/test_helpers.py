import pytest
from datetime import datetime

from src.utilities.helpers import check_final_answer_exist, get_day_date_month_year_time, load_yaml_file


def test_load_yaml_file():
    result = load_yaml_file("src/prompts/instruction.yaml")
    assert "INSTPROMPT" in result


def test_get_day_date_month_year_time():
 
    result = get_day_date_month_year_time()
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)
    assert isinstance(result[2], int)
    assert isinstance(result[3], int)
    assert isinstance(result[4], int)
    assert isinstance(result[5], int)
    assert isinstance(result[6], int)
    assert isinstance(result[7], int)

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