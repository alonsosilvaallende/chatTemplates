# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jinja2",
#     "langchain-core",
#     "protobuf",
#     "pydantic",
#     "sentencepiece",
#     "textual",
#     "transformers",
# ]
# ///

from textual import on
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Input, Label, Static, Switch, Header, Footer, Select
from textual.reactive import reactive
from textual.containers import ScrollableContainer, Horizontal, Container
from transformers import AutoTokenizer


MISTRAL = '{%- if messages[0]["role"] == "system" %}\n    {%- set system_message = messages[0]["content"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}\n            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message["role"] == "user" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- "[AVAILABLE_TOOLS] [" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- \'{"type": "function", "function": {\' }}\n                {%- for key, val in tool.items() if key != "return" %}\n                    {%- if val is string %}\n                        {{- \'"\' + key + \'": "\' + val + \'"\' }}\n                    {%- else %}\n                        {{- \'"\' + key + \'": \' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- ", " }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- "}}" }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- else %}\n                    {{- "]" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- "[/AVAILABLE_TOOLS]" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- "[INST] " + system_message + "\\n\\n" + message["content"] + "[/INST]" }}\n        {%- else %}\n            {{- "[INST] " + message["content"] + "[/INST]" }}\n        {%- endif %}\n    {%- elif message.tool_calls is defined and message.tool_calls is not none %}\n        {{- "[TOOL_CALLS] [" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\n            {%- endif %}\n            {{- \', "id": "\' + tool_call.id + \'"}\' }}\n            {%- if not loop.last %}\n                {{- ", " }}\n            {%- else %}\n                {{- "]" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message["role"] == "assistant" %}\n        {{- " " + message["content"]|trim + eos_token}}\n    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- \'[TOOL_RESULTS] {"content": \' + content|string + ", " }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\n        {%- endif %}\n        {{- \'"call_id": "\' + message.tool_call_id + \'"}[/TOOL_RESULTS]\' }}\n    {%- else %}\n        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}\n    {%- endif %}\n{%- endfor %}\n'

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "NousResearch/Hermes-3-Llama-3.1-8B", "google/gemma-3-1b-it", "Choose a custom model"]

from pydantic import BaseModel, Field

class multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

from langchain_core.utils.function_calling import convert_to_openai_tool

tools = [convert_to_openai_tool(multiply)]

class Name(Widget):
    model = reactive("Qwen/Qwen2.5-0.5B-Instruct")
    hf_key = reactive("")
    system_prompt = reactive("")
    user_prompt = reactive("")
    add_generation_prompt = reactive(False)
    tokenize = reactive(False)
    use_tools = reactive(False)
    enable_thinking = reactive(False)
    representation = reactive(False)

    def render(self) -> str:
        messages = []
        if self.system_prompt != "":
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.user_prompt})
        if self.model == "mistralai/Mistral-7B-Instruct-v0.3":
            model_name = "MaziyarPanahi/Mistral-7B-Instruct-v0.3"
        elif self.model == "google/gemma-3-1b-it":
            model_name = "alonsosilva/GRPOunsloth"
        else:
            model_name = self.model
        try:
            if self.hf_key != "":
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_key)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError as e:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            self.notify(str(e))
        if self.model == "mistralai/Mistral-7B-Instruct-v0.3":
            tokenizer.chat_template = MISTRAL
        if self.use_tools:
            prompt = tokenizer.apply_chat_template(messages, tokenize=self.tokenize, tools=tools, enable_thinking=self.enable_thinking, add_generation_prompt=self.add_generation_prompt)
        else:
            prompt = tokenizer.apply_chat_template(messages, tokenize=self.tokenize, enable_thinking=self.enable_thinking, add_generation_prompt=self.add_generation_prompt)
        if self.representation:
            prompt = repr(prompt)
        if self.tokenize:
            prompt = f"""{prompt}""".replace("[","")
            prompt = f"""{prompt}""".replace("]","")
            return prompt 
        else:
            return f"""{prompt}""".replace("[","\\[")
    

class Watch(Static):
    def compose(self):
        with ScrollableContainer():
            yield Static("I built this tool to help me understand how chat templates work. Choose a model, enter a system prompt and a user prompt, and select different options to see how the input to the LLM changes.", classes="highlight")
            with Horizontal(classes="selectModel"):
                yield Select(((line, line) for line in MODELS), prompt="Select model (required)", value="Qwen/Qwen2.5-0.5B-Instruct")
                yield Container(id="input-container")
                yield Container(id="hf-key-container")
            # yield Input(placeholder="Choose a custom model", id="custom")
            yield Input(placeholder="Write a system prompt (optional)", id="system")
            yield Input(placeholder="Write a user prompt (optional)", id="user")
            with Horizontal(classes="container"):
                yield Static("Add generation prompt:", classes="label")
                focused_switch = Switch(value=False, id="generation")
                focused_switch.focus()
                yield focused_switch
                yield Static("Use tool:", classes="label")
                yield Switch(id="usetool")
                yield Static("Tokenize:", classes="label")
                yield Switch(id="tokenize")
                yield Static("Enable thinking:", classes="label")
                yield Switch(id="think")
                yield Static("String repr:", classes="label")
                yield Switch(id="repr")
            yield Static("What the LLM sees:\n", classes="highlight")
            yield Name()

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if str(event.value) == "Choose a custom model":
            container = self.query_one("#input-container")
            container.query("Input").remove() # Remove any existing input
            container.mount(Input(placeholder="Custom model HuggingFace name", id="custom"))
            container = self.query_one("#hf-key-container")
            container.query("Input").remove() # Remove any existing input
            container.mount(Input(placeholder="HF key (only needed for gated models)", id="hfkey"))
        else:
            self.query_one(Name).model = str(event.value)

    @on(Switch.Changed, "#generation")
    def switch(self, event: Switch.Changed):
        self.query_one(Name).add_generation_prompt = not self.query_one(Name).add_generation_prompt

    @on(Switch.Changed, "#usetool")
    def use_a_tool(self, event: Switch.Changed):
        self.query_one(Name).use_tools = not self.query_one(Name).use_tools

    @on(Switch.Changed, "#tokenize")
    def switch_tokenize(self, event: Switch.Changed):
        self.query_one(Name).tokenize = not self.query_one(Name).tokenize

    @on(Switch.Changed, "#think")
    def switch_enable_thinking(self, event: Switch.Changed):
        self.query_one(Name).enable_thinking = not self.query_one(Name).enable_thinking

    @on(Switch.Changed, "#repr")
    def switch_representation(self, event: Switch.Changed):
        self.query_one(Name).representation = not self.query_one(Name).representation

    @on(Input.Submitted, "#custom")
    def provide_custom_model(self, event: Input.Submitted):
        self.query_one(Name).model = str(event.value)

    @on(Input.Submitted, "#hfkey")
    def provide_hfkey(self, event: Input.Submitted):
        self.query_one(Name).hf_key = str(event.value)

    @on(Input.Submitted, "#system")
    def system(self, event: Input.Submitted):
        self.query_one(Name).system_prompt = event.value

    @on(Input.Submitted, "#user")
    def user(self, event: Input.Submitted):
        self.query_one(Name).user_prompt = event.value

class WatchApp(App):
    CSS ="""
Watch {   
    background: $boost;
    margin: 3 10 5 10;
    min-width: 50;
    padding: 3;
}

.selectModel {
    height: 3;
}

.container {
    height: auto;
    width: auto;
}

.label {
    height: 3;
    content-align: center middle;
    width: auto;
    margin: 0 0 0 1;
}

.highlight {
    color: $primary;
}

Switch {
    border: solid green;
    height: auto;
    width: auto;
}
"""
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield Watch()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = "textual-dark" if self.theme == "textual-light" else "textual-light"

    def on_mount(self) -> None:
        self.title = "Understanding Chat Templates"

if __name__ == "__main__":
    app = WatchApp()
    app.run()
