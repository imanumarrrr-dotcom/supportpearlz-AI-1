from src.chains.schemas import SupportAnswer

def test_refusal_schema():
    response = SupportAnswer(answer="Not documented.", sources=[], confidence="none", answered=False)
    assert response.answered is False
    assert response.confidence == "none"
