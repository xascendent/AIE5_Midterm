import os
import asyncio
import uuid
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from logger import logger
from templates import MetaDataModel
from utils_openai import UtilityOpenAI




async def get_pdf_files(directory: str) -> list[str]:
    """Returns a list of all PDF filenames in the given directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(current_dir, directory)

    if not os.path.exists(pdf_dir):
        logger.error(f"Directory not found: {pdf_dir}")
        return []

    return [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

async def get_pdf_metadata(directory: str, pdf_file: str) -> MetaDataModel:
    """Extracts metadata from a PDF file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_location = os.path.join(current_dir, directory, pdf_file)

    try:
        reader = PdfReader(pdf_location)   
        metadata = reader.metadata

        return MetaDataModel(
                document_name = pdf_file,
                document_id = str(uuid.uuid4()),
                title = metadata.get("/Title", "No title Found"),
                author = metadata.get("/Author", "No author Found"),
                description = metadata.get("/Description", "No description Found"),                
                subject = metadata.get("/Subject", "No subject Found")                
            )
    except Exception as e:
        logger.error(f"Error reading PDF metadata for {pdf_file}: {str(e)}")
        return None
         

# on loading splitting and embedding
#async def load_pdf(directory: str) -> list[Document]:
#    """Loads all PDFs in the given directory."""
#    pdf_files = await get_pdf_files(directory)

#    if not pdf_files:
#        logger.error("No PDF files found.")
#        return []

#    documents = []
#    for file_name in pdf_files:
#        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory, file_name)
#        loader = PyPDFLoader(file_path, extract_images=False)
#        async for page in loader.alazy_load():
#            documents.append(page)

#    return documents

async def load_pdf(directory: str, pdf_file: str) -> list[Document]:
    """Loads a single PDF file and returns a list of Document objects."""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory, pdf_file)

    loader = PyPDFLoader(file_path, extract_images=False)
    documents = []

    async for page in loader.alazy_load():
        documents.append(page)  # Store each page properly

    return documents

async def chunk_pdf_document(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Chunks a list of Document objects into smaller text chunks."""
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    
    # Make sure to pass a list of Document objects, not a single Document
    docs = splitter.split_documents(documents)

    chunks = [doc.page_content for doc in docs]

    logger.debug(f"Pages in the original document: {len(documents)}")
    logger.debug(f"Length of chunks after splitting pages: {len(chunks)}")

    return chunks


async def main():
    print("Ready Player 1")
    dir = "data/pdfs"
    pdf_files = await get_pdf_files(dir)
    utility = UtilityOpenAI()
    
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



if __name__ == "__main__":
    asyncio.run(main())