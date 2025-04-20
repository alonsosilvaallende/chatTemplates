# Chat Templates

Chat is an abstraction. Language models don't know anything about system messages, user messages, tool messages, or any other messages. Language models receive a string representation that is converted to tokens and then sequentially predict the next token (the language model has been fine tuned to react differently when it sees these special tokens). The translation between messages and the received string representation is handled by the chat template of the model (which in many occassions contains errors or does not handle things as one would expect).

I built this tool to help me understand how chat templates work. Select a model or choose a custom model from [HuggingFace](https://huggingface.co/), enter a system prompt and a user prompt, and select different options to see how the input to the LLM changes.


<img width="697" alt="image" src="https://github.com/user-attachments/assets/14c65d15-f4e2-4292-b504-9647bcd44f63" />


To run this app, if you do not have [uv](https://docs.astral.sh/uv/) installed, follow the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to install it. Then run:
```console
uv run https://raw.githubusercontent.com/alonsosilvaallende/chatTemplates/refs/heads/main/chattemplates.py
```
