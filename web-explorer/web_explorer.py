import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from web_search_duckduckgo import WebResearchRetriever
# import WebResearchRetriever

import os
from langchain_community.retrievers import BM25Retriever

# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # Get it at https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials
# os.environ["GOOGLE_CSE_ID"] = "YOUR_CSE_ID" # Get it at https://programmablesearchengine.google.com/
# os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY" # Get it at https://beta.openai.com/account/api-keys


st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")

def settings():

    # Vectorstore
    import faiss
    from langchain.vectorstores import FAISS 
    # from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.embeddings import OllamaEmbeddings 
    from langchain.docstore import InMemoryDocstore  

    model_name, embedding_size = "gemma:7b", 2048
    # model_name, embedding_size = "llama2", 4096

    embeddings_model = OllamaEmbeddings(model=model_name) #OpenAIEmbeddings()  
    # embedding_size = 2048  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM
    from langchain_community.llms import Ollama

    llm = Ollama(model=model_name)#, temperature=0, streaming=True)
    # from langchain.chat_models import ChatOpenAI
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

    # Search
    # from langchain.utilities import GoogleSearchAPIWrapper, DuckDuckGoSearchAPIWrapper
    from langchain.utilities import DuckDuckGoSearchAPIWrapper
    # import googlesearch
    # search = GoogleSearchAPIWrapper()   
    search = DuckDuckGoSearchAPIWrapper(backend = 'api', max_results=5, source = "news", time = 360)
 
    # from google_search_wrapper import GoogleSearchAPIWrapperLocal
    # search = GoogleSearchAPIWrapperLocal(max_results = 2)

    # Initialize 
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm, 
        search=search, 
        num_search_results=5
    )

    return web_retriever, llm

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")
        self.urls = set([])

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        self.urls = set([])
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.urls.add(source)
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)
        # self.container.info('`Sources:`\n\n' + urls)


try:
    st.sidebar.image("./web-explorer/img/ai.png")
except:
    st.sidebar.image("./img/ai.png")
st.header("`Interweb Explorer`")
st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
    "I can be configured to use different modes: public API or private (no data sharing).`")

# Make retriever and llm
if 'retriever' not in st.session_state:
    web_retriever, llm = settings()
#     st.session_state['retriever'], st.session_state['llm'] = settings()
# web_retriever = st.session_state.retriever
# llm = st.session_state.llm

print(f"**** {web_retriever}, {llm}")
# User input 
# question = st.text_input("`Ask a question:`")
question = "how much revenues achieved by Alibaba in 2023"

if question:

    # Generate answer (w/ citations)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Write answer and sources
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])
    answer.info('`Answer:`\n\n' + result['answer'])
    print(f"{result['answer']}")
    st.info('`Sources:`\n\n' + "\n\n".join(retrieval_streamer_cb.urls))
