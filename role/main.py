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
from config import model_cfg, course_cfg

#Set configuration
model_selection = st.sidebar.selectbox("Select LLM", list(model_cfg.model_available))
course_selection = st.sidebar.selectbox("Select training course", list(course_cfg.course_available.keys()))
human_name = st.sidebar.text_input("Set user name", "Mike")

# Clear chat session if dropdown option or radio button changes
if st.session_state.get("current_course") != course_selection or st.session_state.get("current_model") != model_selection or st.session_state.get("current_user") != human_name:
    st.session_state["current_course"] = course_selection
    st.session_state["current_model"] = model_selection
    st.session_state["current_user"] = human_name
    st.session_state["messages"] = [AIMessage(content="Welcome to training course")]

    st.session_state["evaluate"] = []

st.sidebar.markdown(f"""**Course Now**\n\n
    """)
st.sidebar.write(course_cfg.course_available[course_selection]["desc"])
#
ai_persona = course_cfg.course_available[course_selection]["ai_persona"] #"an insurance salesman"
human_persona = course_cfg.course_available[course_selection]["human_persona"].format(human_name) #f""""I am {human_name}""" #, a father  of 2 children, loving adventure."""
company = course_cfg.course_available[course_selection]["company"] #"AIA insurance"

if "_buyer_" in course_selection:
    prompt_template = get_prompt.load_prompt_user_agent(ai_persona, human_role = human_persona, company = company, human_name=human_name, content=None)
else:
    prompt_template = get_prompt.load_prompt(ai_persona, human_role = human_persona, company = company, human_name=human_name, content=None)

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

    model = Ollama(callbacks=callbacks, model=model_selection, temperature=temperature, stop=stop)#, streaming=True)

    return model

# model_name = "llama3:instruct" #"gemma:2b"#"llama2"

try:
    stop_list = model_cfg.param_cfg[model_selection.split(":")[0]]["stop"]
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
    if prompt.lower() != "score":
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            # model = ChatOpenAI(streaming=True, callbacks=[stream_handler], model="gpt-4")
            model = load_ollama(callbacks=[stream_handler], model=model_selection, temperature=0, stop = stop_list)#, streaming=True)

                
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
    else:
        model_eval = load_ollama(model="wizardlm2", temperature=0, stop = model_cfg.param_cfg["wizardlm2"]["stop"])

        docs = []
        for msg in st.session_state["messages"]:
            if isinstance(msg, HumanMessage):
                docs.append("user:"+msg.content)
            else:
                docs.append("assistant:"+msg.content)
        docs = "\n".join(docs)
        print(f"*** docs = {docs}")

        chain_eval = LLMChain(prompt=get_prompt.load_prompt_eval(docs), llm=model_eval)

        response_eval = chain_eval({"input":"", "chat_history":[]}, include_run_info=True)

        st.write(response_eval[chain_eval.output_key])

