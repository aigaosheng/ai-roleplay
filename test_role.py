import streamlit as st
from typing import List
# from langchain.chat_models import ChatVertexAI
from langchain_community.llms import Ollama

model_name = "gemma:2b"
model_name = "llama2"
# ollama_openhermes = Ollama(model=model_name)

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
# from langchain_community.llms import LlamaCpp

ENABLE_UI = False #True

model_name = "/home/gs/hf_home/models/models--google--gemma-2b-it/gemma-2b-it.gguf"
model_name_embed = "/home/gs/hf_home/models/models--google--gemma-2b/gemma-2b.gguf"

from gemma import GemmaLocal, GemmaChatLocal
# from langchain_google_vertexai import GemmaChatLocalHF, GemmaLocalHF

gemma_inst = GemmaChatLocal(model_name = model_name, hf_access_token = "")


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

assistant_role_name = "Mike" #"Individual Insurance Buyer"
user_role_name = "David" #Insurance Sales"
task2 = f"Train pitch skill of {user_role_name}, who is a junior insurance sales in AIA insurance company. He job is to explain the insurance products to the insurance buyers, comminicate with his insurance buyers and answer their questions and concerns. {assistant_role_name} is an individual insurance buyer or a customer of AIA insurance company. {assistant_role_name} is a father of one child and one daughter, and live with his wife. He loves his family and want to buy some insurance products for his family. In the training session, {assistant_role_name} seeks {user_role_name} to help him find suitable insurance products for his family. {assistant_role_name} can ask any questions on AIA insurance products. {user_role_name}, as an insurance sales, must answer {assistant_role_name}'s questions based on his expert knowledge"
word_limit = 100  # word limit for task brainstorming

task = (
    f"""This task is about a conversation between an AIA insurance company's salesperson, {user_role_name}, and a customer, {assistant_role_name}, aimed at training the salesperson's selling skills. I am {user_role_name}, the salesperson of the insurance company, and you, {assistant_role_name}, are my customer. You can ask me any questions about insurance, such as why to buy this insurance, how much the premium is, what is the coverage, how to claim. I must answer all your questions. You are my customer, you can only ask me questions related to insurance, and you cannot answer questions related to insurance. I am your insurance broker, I answer all your questions.
    Backgroud of {assistant_role_name}: You are a software engineer. You married and you and your wife have a 6-year-old son and a 16-year-old daughter. You never bought insurance before, but now you are planning to buy insurance to protect you and your family.
    Background of {user_role_name}: I am a seasoned insurance salesperson, and my job is to answer any questions from customers, ensure customer satisfaction, and persuade them to buy the insurance I recommend.

    """
)


if ENABLE_UI:
    st.title("Langchain-Gemma Agent")
    st.sidebar.header("Input Settings")

    assistant_role_name = st.sidebar.text_input("Assistant Role Name", f"{assistant_role_name}")
    user_role_name = st.sidebar.text_input("User Role Name", f"{user_role_name}")
    task = st.sidebar.text_area("Task", f"{task}")
    word_limit = st.sidebar.number_input("Word Limit for Task Brainstorming", value=word_limit)


# Step-1: Task define agent to generate task description
task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
task_specify_agent = AiAgent(task_specifier_sys_msg, gemma_inst)

task_specifier_prompt = (
    """Here is a task that {assistant_role_name} will collaborate with {user_role_name} to complete the task: {task}.
The training sessision starts with {assistant_role_name}, an insurance customer, to ask {user_role_name}, a insurance sales in AIA, a question about the insurance.

Please make it more specific. Be creative and imaginative.
Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
)
task_specifier_template = HumanMessagePromptTemplate.from_template(template=task) #task_specifier_prompt)
task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                             user_role_name=user_role_name,
                                                             task=task, word_limit=word_limit)[0]
specified_task_msg = task_specifier_msg #task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}")
specified_task = specified_task_msg.content

#Step-2: Define user/assistant agent
assistant_inception_prompt = (
    """Never forget you are {assistant_role_name}, a client, and I am {user_role_name}, an insurance salesperson. Never flip roles! 
Here is the task: {task}. Never forget our task!
I must give you one answer at a time.
You must write a specific question.
You are supposed to ask me any questions.
Unless I say the task is completed, you should always start with:

Solution: <YOUR_QUESTION>

Always end <YOUR_QUESTION> with: Next request."""
)

user_inception_prompt = (
    """Never forget you are {user_role_name}, an insurance salesperson. I am {assistant_role_name}, a client. Never flip roles! 
Here is the task: {task}. Never forget our task!
You must answer my question to complete the task ONLY in the following way:

Answer: <YOUR_ANSWER>

You must give me one answer at a time.
I must write a response that appropriately completes the requested question.
Keep giving me answer and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task."""
)


assistant_inception_prompt2 = (
    """Never forget you are {assistant_role_name} and I am {user_role_name}. Never flip roles!

We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {task}. Never forget our task!

I must solve your problem or concern and complete the task.
You must ask me one question about AIA insurance product.

Unless I say the task is completed, you should always start with:

Question: <YOUR_QUESTION>

<YOUR_QUESTION> should be specific on the insurance product.
Always end <YOUR_QUESTION> with: Next request."""

)

user_inception_prompt2 = (
    """Never forget you are {user_role_name}, and I am {assistant_role_name}. Never flip roles! 

We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: {task}. Never forget our task!
You must solve my question or concern about AIA insurance product.

I must write a response that appropriately expresses question or concern on the product.

I answer your question to complete the task ONLY in the following way:
Answer: <MY_ANSWER>.

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
assistant_agent = AiAgent(assistant_sys_msg, gemma_inst)
user_agent = AiAgent(user_sys_msg, gemma_inst)

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats
assistant_msg = HumanMessage(
    content=(f"{user_sys_msg.content}. "
             f"Hello."
            ))

# user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
# user_msg = assistant_agent.step(user_msg)

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
