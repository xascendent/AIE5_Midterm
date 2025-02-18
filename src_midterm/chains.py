import os
import datetime
from enum import Enum
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI # might need to remove if we create our own model

load_dotenv()

class LLMToUse(Enum):
    gpt_4o_mini = "gpt-4o-mini"
    LLAMA_3_2 = "llama3.2"

# Initialize the LLM choice
llm_to_use = LLMToUse.OPEN_AI_MINI


ot_researcher_prompt = ChatPromptTemplate.from_messages(
    [
    (
        "system",
        """You are a expert occupational therapist researcher. 
        Current time {time}
        1. {first_instruction}
        2. Reflect and critique your answer.  Be sever to maximize imporvement.
        3. Recommend search quries to research information to imporve your answer."""

    ),
    MessagesPlaceholder(variable_name="messages"), # this is the placeholder for historical messages
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)


def init_model(llm_choice):
    if llm_choice == LLMToUse.OPEN_AI_MINI:
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

ot_research_chain = ot_researcher_prompt | init_model(llm_to_use)