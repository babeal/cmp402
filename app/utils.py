from typing import Dict, List, Sequence

import boto3
import logging

from langchain.schema.runnable import ConfigurableField
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from lib.langchain.llama2 import Llama2Chat
from lib.langchain.opensearch import create_ovs_client
from langchain.schema import Document
from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
)

logger = logging.getLogger(__name__)


def get_bedrock_embeddings(boto3_session: boto3.Session) -> BedrockEmbeddings:
    bedrock_client = boto3_session.client("bedrock-runtime")
    embeddings_model_id = "amazon.titan-embed-text-v1"
    return BedrockEmbeddings(client=bedrock_client, model_id=embeddings_model_id)


def get_opensearch_vector_store(
    boto3_session: boto3.Session, bedrock_embeddings: BedrockEmbeddings
) -> OpenSearchVectorSearch:
    aoss_client = boto3_session.client("opensearchserverless")
    list_collections_response = aoss_client.list_collections()
    collection_id = list_collections_response.get("collectionSummaries")[0].get("id")
    index_name = "bedrock-docs"
    return create_ovs_client(
        collection_id,
        index_name,
        boto3_session.region_name,
        boto3_session,
        bedrock_embeddings,
    )


def get_llama2chat(
    boto3_session: boto3.Session, streaming=False, endpoint_name="llama-2-7b"
) -> Llama2Chat:
    return Llama2Chat(
        endpoint_name=endpoint_name,
        client=boto3_session.client("sagemaker-runtime"),
        streaming=streaming,
    ).configurable_fields(
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        ),
        top_p=ConfigurableField(
            id="top_p",
            name="LLM Top P",
            description="The top p of the LLM",
        ),
        max_tokens_to_sample=ConfigurableField(
            id="max_tokens_to_sample",
            name="LLM Max Tokens",
            description="The max tokens to create",
        ),
    )


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("role") == "user":
            converted_chat_history.append(HumanMessage(content=message.get("content")))
        if message.get("role") == "assistant":
            converted_chat_history.append(AIMessage(content=message.get("content")))

    return converted_chat_history


def escape_content(content):
    content = content.replace(":", "\:")
    return content
