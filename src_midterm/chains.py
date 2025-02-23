import os
import datetime
from enum import Enum
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI # might need to remove if we create our own model
from logger import logger

load_dotenv()

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
        """
    ),
    MessagesPlaceholder(variable_name="messages"), # this is the placeholder for historical messages
    ]
)

ot_researcher_prompt = ChatPromptTemplate.from_messages(
    [
    (
        "system",
        """You are a expert occupational therapist researcher. 
        Current time {time}
        1. Reflect and critique your answer.  Be sever to maximize imporvement.
        2. Recommend search quries to research information to imporve your answer."""

    ),
    MessagesPlaceholder(variable_name="messages"), # this is the placeholder for historical messages
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

ot_post_process_prompt = ChatPromptTemplate.from_messages(
    [
    (
        "system",
        """You are an expert at enchancing the quality of the answer.  Keep it subtle and do not alter the content or meaning of the answer.
        """
    ),
    MessagesPlaceholder(variable_name="messages"), # this is the placeholder for historical messages
    ]
)


def init_model(llm_choice):
    if llm_choice == LLMToUse.gpt_4o_mini:
        return use_openai_chain()
    elif llm_choice == LLMToUse.LLAMA_3_2:
        return use_llama_chain()
    else:
        raise ValueError("Unsupported LLM choice")
    
def use_openai_chain():
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
    return llm

def use_llama_chain():   
    pass

ot_user_chain = ot_user_prompt | init_model(llm_to_use)
ot_research_chain = ot_researcher_prompt | init_model(llm_to_use)
ot_post_process_chain = ot_post_process_prompt | init_model(llm_to_use)