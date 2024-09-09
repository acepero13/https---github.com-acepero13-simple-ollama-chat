from flask import Flask, request, jsonify
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any

app = Flask(__name__)

class OllamaChat:
    def __init__(self):
        self.chat_model = None

    def chat(self, messages: List[Dict[str, str]], model_name: str) -> Dict[str, Any]:
        if self.chat_model is None:
            self.chat_model = ChatOllama(model=model_name)
        
        langchain_messages = []
        for message in messages:
            if message['role'] == 'system':
                langchain_messages.append(("system", message['content']))
                #langchain_messages.append(SystemMessage(content=message['content']))
            elif message['role'] == 'user':
                langchain_messages.append(("human", message['content']))
                #langchain_messages.append(HumanMessage(content=message['content']))
            elif message['role'] == 'assistant':
                langchain_messages.append(("assistant", message['content']))
                #langchain_messages.append(AIMessage(content=message['content']))
            else:
                return {'error': f"Unsupported message role: {message['role']}"}
        
        try:
            response = self.chat_model.invoke(langchain_messages)
            print(response)
            return response.content
        except Exception as e:
            return {'error': str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        if self.chat_model:
            return {
                'model_name': self.chat_model.model,
                'provider': 'Ollama',
            }
        else:
            return {'error': 'Model not initialized'}

ollama_chat = OllamaChat()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'messages' not in data or 'model' not in data:
        return jsonify({'error': 'Invalid request. Please provide a list of messages and model name.'}), 400

    messages = data['messages']
    model_name = data['model']
    response = ollama_chat.chat(messages, model_name)
    return jsonify(response)

@app.route('/model-info', methods=['GET'])
def model_info():
    info = ollama_chat.get_model_info()
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5006)
