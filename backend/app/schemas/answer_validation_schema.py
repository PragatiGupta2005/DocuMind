from pydantic import BaseModel


class AnswerValidationResult(BaseModel):
    is_grounded: bool
    reason: str