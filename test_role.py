import streamlit as st
from typing import List
# from langchain.chat_models import ChatVertexAI

from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    
    BaseMessage,
)
from langchain_community.llms import LlamaCpp

ENABLE_UI = False

model_name = "/home/gs/hf_home/models/models--google--gemma-2b-it/gemma-2b-it.gguf"
model_name_embed = "/home/gs/hf_home/models/models--google--gemma-2b/gemma-2b.gguf"

from gemma import GemmaLocal, GemmaChatLocal
# from langchain_google_vertexai import GemmaChatLocalHF, GemmaLocalHF

# ChatOpenAI = GemmaChatLocal(model_name = model_name, hf_access_token = "")


class AiAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: GemmaChatLocal,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)#s[-1].content)
        self.update_messages(output_message)

        return output_message

assistant_role_name = "Customer"
user_role_name = "Sale"
task = f"Simulate the pitch training session in the insurance company to improve {user_role_name}'s pitch skill to sell insurance product to {assistant_role_name}."
word_limit = 100  # word limit for task brainstorming

if ENABLE_UI:
    st.title("Langchain-Gemma Agent")
    st.sidebar.header("Input Settings")

    assistant_role_name = st.sidebar.text_input("Assistant Role Name", f"{assistant_role_name}")
    user_role_name = st.sidebar.text_input("User Role Name", f"{user_role_name}")
    task = st.sidebar.text_area("Task", f"{task}")
    word_limit = st.sidebar.number_input("Word Limit for Task Brainstorming", value=word_limit)


# Step-1: Task define agent to generate task description
task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
task_specify_agent = AiAgent(task_specifier_sys_msg, GemmaChatLocal(model_name = model_name, hf_access_token = "",temperature=1.0, allow_reuse = True))

task_specifier_prompt = (
    """Here is a task that {assistant_role_name} will help {user_role_name} to complete the task: {task}.
The training sessision starts with {assistant_role_name} to enquiry products sold by {user_role_name}. 
Please make it more specific. Be creative and imaginative.
Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
)
task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                             user_role_name=user_role_name,
                                                             task=task, word_limit=word_limit)[0]
specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}")
specified_task = specified_task_msg.content

#Step-2: Define user/assistant agent
assistant_inception_prompt = (
    """Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles! 
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {task}. Never forget our task!
I must rightly address your question and my needs to complete the task.

You must simulate the real insurance customer and enquiry me one question.
Unless I say the task is completed, you should always start with:

Question: <YOUR_QUESTION>

<YOUR_QUESTION> should be specific on the insurance product and concerns on the product.
Always end <YOUR_QUESTION> with: Next request."""

)

user_inception_prompt = (
    """Never forget you are {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always question me.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: {task}. Never forget our task!
You must ask me on the insurance product.

You must ask me one question at a time.
I must write a response that appropriately completes the requested question.

When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task."""
)

def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = \
        assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name,
                                               task=task)[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = \
        user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name,
                                          task=task)[0]

    return assistant_sys_msg, user_sys_msg

# Create AI assistant agent and AI user agent from obtained system messages
assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
assistant_agent = AiAgent(assistant_sys_msg, GemmaChatLocal(model_name = model_name, hf_access_token = "",temperature=0.2, allow_reuse = True))
user_agent = AiAgent(user_sys_msg,  GemmaChatLocal(model_name = model_name, hf_access_token = "",temperature=0.25, allow_reuse = True))

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats
assistant_msg = HumanMessage(
    content=(f"{user_sys_msg.content}. "
             "Now start to give me enquiries one by one. "
            ))

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
user_msg = assistant_agent.step(user_msg)

if ENABLE_UI:
    st.header("Conversation")

chat_turn_limit, n = 5, 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    # print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")

    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    # print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")

    # Display the conversation in chat format
    if ENABLE_UI:
        st.text(f"AI User ({user_role_name}):")
        st.info(user_msg.content)
        st.text(f"AI Assistant ({assistant_role_name}):")
        st.success(assistant_msg.content)
        if "<CAMEL_TASK_DONE>" in user_msg.content:
            break

# task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
# task_specifier_prompt = """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
# Please make it more specific. Be creative and imaginative.
# Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
# task_specifier_template = HumanMessagePromptTemplate.from_template(
#     template=task_specifier_prompt
# )
# task_specify_agent = AiAgent(task_specifier_sys_msg, ChatOpenAI)#(temperature=1.0))
# task_specifier_msg = task_specifier_template.format_messages(
#     assistant_role_name=assistant_role_name,
#     user_role_name=user_role_name,
#     task=task,
#     word_limit=word_limit,
# )[0]
# specified_task_msg = task_specify_agent.step(task_specifier_msg)
# print(f"Specified task: {specified_task_msg.content}")
# specified_task = specified_task_msg.content
