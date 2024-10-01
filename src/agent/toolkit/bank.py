from typing import Optional
from langchain.tools import Tool
from pydantic.v1 import BaseModel, root_validator


class BankAccount(BaseModel):
    bank_api_key: Optional[str] = None

    def run(self, query: str):
        # TODO: implement an API
        return "Current Account Balance: $25,757.57"

    async def arun(self, query: str):
        return "Current Account Balance: $25,757.57"


def get_bank_account(name: str, description: str):
    bankAccount = BankAccount()
    return Tool(
        name=name,
        description=description,
        func=bankAccount.run,
        coroutine=bankAccount.arun,
    )
