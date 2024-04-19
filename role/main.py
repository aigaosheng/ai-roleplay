from langchain.callbacks.base import BaseCallbackHandler
# from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama

from langchain.schema import HumanMessage, AIMessage
import streamlit as st

import os

try:
    from langsmith import Client
    client = Client()
except:
    client = None


st.set_page_config(page_title="Virtual Coaching and Training", page_icon="🦜")
st.title("🦜 Coaching and Training")
button_css =""".stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 2em;
    width: 2em;
    font-size: 4px;
}"""
st.markdown(f'<style>{button_css}</style>', unsafe_allow_html=True)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

try:
    with open("./guide.txt", "r") as f:
        guide = f.read()
except:
    guide = None

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from insurance import get_prompt
from config import model_cfg

ai_persona = "an insurance salesman"
human_persona = """"I am Mike, a father  of 2 children, loving adventure."""
company = "AIA insurance"

prompt_template = get_prompt.load_prompt(ai_persona, human_role = human_persona, company = company, content=None)

from langchain.chains import LLMChain


def send_feedback(run_id, score):
    if client:
        client.create_feedback(run_id, "user_score", score=score)
    else:
        return None

def load_ollama(**kwargs):
    model_name = kwargs.get("model", "llama3:instruct")
    temperature = kwargs.get("temperature", 0.0)
    # streaming = kwargs.get("streaming", True)
    callbacks = kwargs.get("callbacks", [])
    stop = kwargs.get("stop", [])

    model = Ollama(callbacks=callbacks, model=model_name, temperature=temperature, stop=stop)#, streaming=True)

    return model

model_name = "llama3:instruct" #"gemma:2b"#"llama2"

try:
    stop_list = model_cfg.param_cfg[model_name.split(":")[0]]["stop"]
except:
    stop_list = None

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content=f"Welcome to {company}? How can I help you")]

for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        # model = ChatOpenAI(streaming=True, callbacks=[stream_handler], model="gpt-4")
        model = load_ollama(callbacks=[stream_handler], model=model_name, temperature=0, stop = stop_list)#, streaming=True)

        
        
        chain = LLMChain(prompt=prompt_template, llm=model)
        print(f"*** prompt_template = {prompt_template}")

        response = chain({"input":prompt, "chat_history":st.session_state.messages[-20:]}, include_run_info=True)
        print(f"*** {st.session_state.messages}")
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.messages.append(AIMessage(content=response[chain.output_key]))
        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2,1,1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍")#, on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎")#, on_click=send_feedback, args=(run_id, 0))

