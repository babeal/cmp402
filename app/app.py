import streamlit as st
import boto3
from prompts import SYSTEM_TEMPLATE, REPHRASE_TEMPLATE
from utils import (
    get_bedrock_embeddings,
    get_llama2chat,
    get_opensearch_vector_store,
    format_docs,
    serialize_history,
    escape_content,
)
from lib.langchain.llama2 import Llama2Chat
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    Runnable,
    RunnablePassthrough,
    RunnableLambda,
    RunnableMap,
    RunnableBranch,
)
from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
)

# Set streaming to True to use the streaming endpoint
streaming = True

# Initialize services
# - Bedrock Embeddings
# - OpenSearch Vector Store
# - Llama2Chat llm
boto3_session = boto3.Session()
bedrock_embeddings = get_bedrock_embeddings(boto3_session)
opensearch_vector_store = get_opensearch_vector_store(boto3_session, bedrock_embeddings)
llama2_chat_model = get_llama2chat(boto3_session, streaming=streaming)


def get_rag_chat_chain(
    llm: Llama2Chat,
    vector_store: OpenSearchVectorSearch,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens_to_sample: int = 100,
    num_documents_to_return: int = 2,
):
    """
    Returns a retrieval augmented generation chat chain that can be used to generate responses to user input
    using data retrieved from a vector store.

    Args:
        llm (Llama2Chat): The Llama2Chat model to use for generating responses.
        retriever: The retriever to use for retrieving relevant documents.
        temperature (float, optional): The temperature to use for sampling from the model. Defaults to 0.6.
        top_p (float, optional): The top_p value to use for sampling from the model. Defaults to 0.9.
        max_tokens_to_sample (int, optional): The maximum number of tokens to sample from the model. Defaults to 100.
        num_documents_to_return (int, optional): The maximum number of documents to retrieve from the vector store. Defaults to 2.

    Returns:
        Runnable: A RAG chat chain that can be used to generate responses to user input.
    """

    # Set configurable hyperparameters for the model
    chat_llm = llm.with_config(
        configurable={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens_to_sample": max_tokens_to_sample,
        }
    )

    input_map = {
        "question": RunnableLambda(itemgetter("question")),
        "chat_history": RunnableLambda(serialize_history),
    }

    ######################################################
    # ADD CODE HERE
    ######################################################

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    chat_retrieval_chain = input_map | rag_prompt | chat_llm | StrOutputParser()

    return chat_retrieval_chain


st.set_page_config(page_title="RAG Chatbot")

# initialize chat history session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(escape_content(message["content"]))


# clear chat history
def clear_chat_history():
    st.session_state.messages = []


# Add the sidebar for the model parameters to the app
with st.sidebar:
    st.image("./assets/logo_light.svg", width=250)  # page logo

    st.subheader("Model parameters")
    # Add model parameter inputs
    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01
    )
    top_p = st.sidebar.slider(
        "top_p", min_value=0.01, max_value=1.0, value=0.90, step=0.01
    )
    max_tokens_to_sample = st.sidebar.slider(
        "max_tokens_to_sample", min_value=32, max_value=512, value=300, step=8
    )

    st.subheader("Retriever parameters")
    # Add retriever parameter inputs
    num_documents_to_return = st.sidebar.slider(
        "num_documents_to_return", min_value=1, max_value=3, value=2, step=1
    )

    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

# Add a chat input to the app
# On send write the message to the to the messages in session state
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(escape_content(prompt))

# Add a session state messages handler for when a new user message is added
# When a new user message is added, call the LLM and update the messages history
if len(st.session_state.messages) and st.session_state.messages[-1]["role"] not in [
    "assistant",
    "system",
]:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # create chat chain
            chain = get_rag_chat_chain(
                llm=llama2_chat_model,
                vector_store=opensearch_vector_store,
                temperature=temperature,
                top_p=top_p,
                max_tokens_to_sample=max_tokens_to_sample,
                num_documents_to_return=num_documents_to_return,
            )
            # prepare chat history by removing the user question and limiting
            # history to the last 2 exchanges since we are limited in context length.
            # LangChain has many classes to assist with conversation history management
            # for production applications.
            chat_history = st.session_state.messages[:-1][-4:]

            placeholder = st.empty()
            full_response = ""
            if streaming:
                response = chain.stream(
                    {"chat_history": chat_history, "question": prompt}
                )
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
            else:
                response = chain.invoke(
                    {"chat_history": chat_history, "question": prompt}
                )
                full_response = response

            placeholder.markdown(escape_content(full_response))
    # add the response from the model to the chat history
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
