from pydantic import BaseModel, Field
from typing import List, Dict
import os
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import GoogleGenerativeAI
from app.services.logger import setup_logger

logger = setup_logger()

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()
    
class EssayGradingAssistant:
    def __init__(self, rubrics, prompt=None, model=None, parser=None, verbose=False):
        default_config = {
            "model": GoogleGenerativeAI(model="gemini-1.5-flash"),
            "parser": JsonOutputParser(pydantic_object=EssayGradeOutput),
            "prompt": read_text_file("prompt/essay-generating-assistant-prompt.txt"),
        }

        self.rubrics = rubrics
        self.prompt = prompt or default_config["prompt"]
        self.model = model or default_config["model"]   
        self.parser = parser or default_config["parser"]
        self.verbose = verbose
    
    def compile(self):
        # Compile the chain here
        pass

    def grade_essay(self, docs: Document):
        # Generate the grade and feedback for the essay
        pass

class EssayGradeOutput(BaseModel):
    grade: float=Field(description="The grade of the essay provided.")
    feedback: str=Field(description="The feedback for the essay provided.")