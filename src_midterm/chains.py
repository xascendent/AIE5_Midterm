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

summarization_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Summarize the following document while keeping all relevant details. Be concise but do not alter the meaning.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given the provided summary, answer the user's query with evidence-based accuracy.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

format_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given this data I want you to break out the response into this format and add why this is good information to provide to the user in the section: 
            1. **Eccentric Exercises**: 

            2. **Isometric Exercises**: 

            3. **Stretching**: 

            4. **Manual Therapy**: 

            5. **Ultrasound Therapy**: 

            6. **Taping and Bracing**: 

            7. **Functional Activities**: 

            8. **Other**:

            9. **Document Title**:

            10. **Document File Name**:
            
            I do not want you to add any additional information and if you don't have the information for the specific section, add I do not have information for this section.  Other will capture 
            any other information that does not fit into the other categories.  Please make sure to add the document title and document file name at the end of the response.
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
summarization_llm = init_model(llm_to_use)

# Define user chain with the LLM
ot_user_chain = ot_user_prompt | llm_instance
summarization_chain = summarization_prompt | summarization_llm
final_chain = final_prompt | llm_instance
format_final_chain = format_final_prompt | llm_instance


async def run_test_query(user_question: str):
    """Runs a test query through the pipeline"""
    reconstructed_document_file_name = ""
    reconstructed_document_title = ""

    # Step 1: Retrieve similar documents from Qdrant
    search_results = search_qdrant(user_question)

    summary = None  # Default summary

    if search_results and search_results[0]["score"] > 0.5:
        pdf_file = search_results[0]["metadata"]["document_name"]
        reconstructed_document_title = search_results[0]["metadata"]["title"]
        reconstructed_document_file_name = pdf_file
        document_text = await get_document(pdf_file)  
        
        # Reconstruct document from chunks
        reconstructed_document = "".join(file.page_content for file in document_text)

        # Step 2: Summarize the document
        summary = summarization_chain.invoke(
            {"messages": [{"role": "system", "content": reconstructed_document}]}
        )
        # Add the document title to the summary
        summary = f" DOCUMENT_TITLE: {reconstructed_document_title}: {summary}"
        summary = f" DOCUMENT_FILE_NAME: {reconstructed_document_file_name}: {summary}"

    # Step 3: Inject the results and user query as context
    context_messages = [{"role": "user", "content": user_question}]

    if summary:
        context_messages.insert(0, {"role": "system", "content": f"Summary of relevant document: {summary}"})


    response = final_chain.invoke({"messages": context_messages})  # This is an AIMessage object

    # Step 4: Convert AIMessage to string before formatting
    formatted_response = format_final_chain.invoke({"messages": [{"role": "system", "content": response.content}]})

    return formatted_response



    


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
        vectors = utility.create_embeddings_from_text(chunks)
         
        qdrant.insert_documents(COLLECTION_NAME, vectors, metadata.to_dict())  


async def main():
    print("\n🚀 Ready player one!\n")  
    COLLECTION_NAME = "qt_document_collection"

    # Initialize OpenAI Utility
    utility = UtilityOpenAI()
    embedding_dim = utility.get_embedding_dimension()

    # Initialize Qdrant
    qdrant = UtilityQdrant(COLLECTION_NAME, embedding_dim)    

    await load_documents(COLLECTION_NAME, utility)

    query = "What specific therapeutic activities and exercises have been shown to be most effective in resolving symptoms and treating chronic tennis elbow"

    response = await run_test_query(query)

    # ✅ Formatting output for readability
    print("\n📌 **Formatted Response**\n")
    print("-" * 50)
    print(response.content)  # Assuming response.content is the final output text
    print("-" * 50)



if __name__ == "__main__":
    asyncio.run(main())
