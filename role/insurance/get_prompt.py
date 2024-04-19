from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

def load_prompt(ai_role, human_role = None, company = None, human_name = "Mike", content = None):
	
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

    Please remember my name is {human_name}.

    Do not output the tone words.

	{content}

	Now remember short response with 1-3 sentences.""".format(content=content, ai_role=ai_role, human_role=human_role,company=company, human_name=human_name)
	

    prompt_template = ChatPromptTemplate(messages = [
		SystemMessage(content=template), 
		MessagesPlaceholder(variable_name="chat_history"), 
		HumanMessagePromptTemplate.from_template("{input}")
		]
    )
    
    return prompt_template

def load_prompt_user_agent(ai_role, human_role = None, company = None, human_name = "David", content = None):
	
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

    template = """I want you to act as {human_role}. Try to ask something to me. I am {ai_role} {company}. I try to market the product to you and make it look more valuable than it is and convince you to buy it. Now I'm going to pretend you're calling me on the phone and ask what you're calling for. Hello, what did you call for?

	This is an interactive conversation between you and me - try you best to query anything on the product in {company} and guide me!

    Please remember my name is {human_name}.

    Do not output the tone words.

	{content}

	Now remember short response with 1-3 sentences.""".format(content=content, ai_role=ai_role, human_role=human_role,company=company, human_name=human_name)
	

    prompt_template = ChatPromptTemplate(messages = [
		SystemMessage(content=template), 
		MessagesPlaceholder(variable_name="chat_history"), 
		HumanMessagePromptTemplate.from_template("{input}")
		]
    )
    
    return prompt_template

def load_prompt_eval(content):
	
    template = """I want you to act as a teacher. Your task is to evaluate the quality between the AI assistant and user, and score the quality in terms of scale 1 to 5. 1 is lowest and 5 is highest. 
    
    The conversation is in the following document.

    -------------------

	{content}

    ---------------------

    The ouitput format is: 
    Training score: 

    """.format(content=content)
	

    prompt_template = ChatPromptTemplate(messages = [
		SystemMessage(content=template), 
		MessagesPlaceholder(variable_name="chat_history"), 
		HumanMessagePromptTemplate.from_template("{input}")
		]
    )
    
    return prompt_template
