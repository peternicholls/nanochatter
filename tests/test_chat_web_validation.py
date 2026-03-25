from fastapi import HTTPException

from scripts.chat_web import ChatMessage, ChatRequest, validate_chat_request


def test_validate_chat_request_accepts_user_and_assistant_roles():
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
        ],
        temperature=0.5,
        max_tokens=32,
        top_k=10,
    )

    validate_chat_request(request)


def test_validate_chat_request_rejects_system_role_with_supported_role_message():
    request = ChatRequest(messages=[ChatMessage(role="system", content="hidden prompt")])

    try:
        validate_chat_request(request)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "Message 0 has invalid role. Must be 'user' or 'assistant'"
    else:
        raise AssertionError("Expected HTTPException for unsupported role")