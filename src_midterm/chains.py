import os
import datetime
from enum import Enum
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI  # Might need to remove if we create our own model
from logger import logger
from qdrant import UtilityQdrant
from utils_openai import UtilityOpenAI
from langchain_core.tools import tool
from typing import Annotated, Dict, List, Tuple, Union
import asyncio
from document_loader import get_pdf_files, chunk_pdf_document, get_pdf_metadata, load_pdf

load_dotenv()

COLLECTION_NAME = "qt_document_collection"
dir = "data/pdfs"
# Initialize OpenAI Utility
utility = UtilityOpenAI()
embedding_dim = utility.get_embedding_dimension()

# Initialize Qdrant
qdrant = UtilityQdrant(COLLECTION_NAME, embedding_dim)


class LLMToUse(Enum):
    gpt_4o_mini = "gpt-4o-mini"
    LLAMA_3_2 = "llama3.2"


# Initialize the LLM choice
llm_to_use = LLMToUse.gpt_4o_mini


ot_user_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an occupational therapist providing accurate, evidence-based answers.
            1. Only give correct information.
            2. If unsure, respond with: "I don't know."
            3. Be clear, concise, and helpful.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

@tool
def search_qdrant(query: Annotated[str, "query to ask the retrieve information tool"]):
    """Search Qdrant for similar documents."""
    query_vector = utility.create_embeddings_from_text([query])[0]
    results = qdrant.search(COLLECTION_NAME, query_vector, 3)
    return results  # Returns retrieved documents

@tool
async def get_document(document_name: Annotated[str, "retrieve the complete document tool"]):
    """Retrieve the complete document."""
    document = await load_pdf(dir, document_name)
    return document


def init_model(llm_choice):
    if llm_choice == LLMToUse.gpt_4o_mini:
        return use_openai_chain()
    elif llm_choice == LLMToUse.LLAMA_3_2:
        return use_llama_chain()
    else:
        raise ValueError("Unsupported LLM choice")


def use_openai_chain():
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(api_key=api_key, model="gpt-4o-mini")


def use_llama_chain():
    pass


# Initialize the LLM
llm_instance = init_model(llm_to_use)

# Define user chain with the LLM
ot_user_chain = ot_user_prompt | llm_instance


async def run_test_query(user_question: str):
    """Runs a test query through the pipeline"""
    dir = "data/pdfs"

    # Step 1: Retrieve similar documents from Qdrant
    search_results = search_qdrant(user_question)
    
    print('------------------------------')
    print(f"document Results:{search_results}")
    print('------------------------------')



    pdf_file = search_results[0]["metadata"]["document_name"]
    document = await load_pdf(dir, pdf_file)  

    
    # Step 2: Inject the results as context
    response = ot_user_chain.invoke(
        {"messages": [{"role": "user", "content": user_question}, {"role": "system", "content": str(search_results)}]}
    )

    return response


async def load_documents(COLLECTION_NAME: str, utility: UtilityOpenAI):
    dir = "data/pdfs"

    pdf_files = await get_pdf_files(dir)
    for pdf_file in pdf_files:
        logger.debug(f"Processing PDF: {pdf_file}")

        documents = await load_pdf(dir, pdf_file)  
        if not documents:
            logger.error(f"Failed to load {pdf_file}")
            continue

        chunks = await chunk_pdf_document(documents) 
        metadata = await get_pdf_metadata(dir, pdf_file)

        logger.debug(f"Metadata: {metadata}")
        logger.debug(f"First chunk: {chunks[0] if chunks else 'No chunks generated'}")
        vectors = utility.create_embeddings_from_text(chunks)
        logger.debug(f"Number of vectors: {len(vectors)}")
        logger.debug(f"First vector: {vectors[0]}")        
        qdrant.insert_documents(COLLECTION_NAME, vectors, metadata.to_dict())  


async def main():

    print("Ready player one")  
    COLLECTION_NAME = "qt_document_collection"
    # Initialize OpenAI Utility
    utility = UtilityOpenAI()
    embedding_dim = utility.get_embedding_dimension()
    # Initialize Qdrant
    qdrant = UtilityQdrant(COLLECTION_NAME, embedding_dim)    
    
    await load_documents(COLLECTION_NAME, utility)

    query = "What specific therapeutic activities and exercises have been shown to be most effective in resolving symptoms and treating chronic tennis elbow"
    print(await run_test_query(query))


if __name__ == "__main__":
    asyncio.run(main())




