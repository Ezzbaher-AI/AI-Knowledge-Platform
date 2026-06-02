conversation_memory = {}


def create_chat(user_id, chat_id):
    if user_id not in conversation_memory:
        conversation_memory[user_id] = {}

    if chat_id not in conversation_memory[user_id]:
        conversation_memory[user_id][chat_id] = []


def save_message(user_id, chat_id, role, content):
    create_chat(user_id, chat_id)

    conversation_memory[user_id][chat_id].append({
        "role": role,
        "content": content
    })


def get_history(user_id, chat_id):
    create_chat(user_id, chat_id)
    return conversation_memory[user_id][chat_id]


def get_last_assistant_message(user_id, chat_id):
    history = get_history(user_id, chat_id)

    for msg in reversed(history):
        if msg["role"] == "assistant":
            return msg["content"]

    return None


def list_user_chats(user_id):
    if user_id not in conversation_memory:
        return []

    return list(conversation_memory[user_id].keys())