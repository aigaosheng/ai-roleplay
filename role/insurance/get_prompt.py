from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

def load_prompt(ai_role, human_role = None, company = None, content = None):
	
    if content:
        content = """
        If I ask questions not related to the following supported document, \
        you should politely decline to answer and remind me to stay on topic.\
        -----------------

        {content}
        
        -----------------
        End of Content.

        """.format(content = content)
    
    else:
        content = ""
    
    human_role = human_role if human_role else ""
    company = f"in {company}"  if company else ""

    template = """{human_role} I want you to act as {ai_role} {company}. Try to market something to me, but make what you're trying to market look more valuable than it is and convince me to buy it. Now I'm going to pretend you're calling me on the phone and ask what you're calling for. Hello, what did you call for?

	This is an interactive conversation between you and me - try you best to engage and guide me along using professional tone!

    Please remember my name is Mike.

    Do not output the tone words.

	{content}

	Now remember short response with 1-3 sentences.""".format(content=content, ai_role=ai_role, human_role=human_role,company=company)
	

    template2 = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

    ### Instruction:
    {instruction}

    ### Input: 
    {input}

    ### Response: """


    prompt_template = ChatPromptTemplate(messages = [
		SystemMessage(content=template), 
		MessagesPlaceholder(variable_name="chat_history"), 
		HumanMessagePromptTemplate.from_template("{input}")
		]
    )
    
    return prompt_template