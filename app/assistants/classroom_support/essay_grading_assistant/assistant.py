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
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["context", "rubrics", "chat_history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        chain = prompt | self.model | self.parser

        return chain

    def grade_essay(self, docs: Document):
        # Generate the grade and feedback for the essay
        pass

class EssayGradeOutput(BaseModel):
    grade: float=Field(description="The grade of the essay provided.")
    feedback: str=Field(description="The feedback for the essay provided.")
    model_config = {
        "json_schema_extra": {
            "example": {
                "rubrics": {
                    "Introduction": "The introduction should be clear and concise.",
                    "Body": "The body should be well-structured and provide supporting evidence.",
                    "Conclusion": "The conclusion should summarize the main points of the essay.",
                },
                "context": """
                    In today's rapidly evolving world, technology plays an increasingly vital role in our daily lives. This essay explores the impact of digital transformation on society. The widespread adoption of smartphones and internet connectivity has fundamentally changed how we communicate, work, and learn.
                    First, the digital revolution has transformed the workplace. Remote work capabilities have become essential, allowing businesses to operate globally. Cloud computing and collaborative tools enable teams to work efficiently across different time zones and locations. This shift has led to increased productivity and work-life balance for many professionals.
                    Furthermore, education has been revolutionized through online learning platforms and digital resources. Students now have access to vast knowledge databases and interactive learning tools. Virtual classrooms have made education more accessible to people worldwide, breaking down geographical barriers and democratizing learning opportunities.
                    In conclusion, digital transformation has reshaped our society in profound ways. The integration of technology in our daily routines has created new opportunities while presenting unique challenges. As we continue to adapt to these changes, it's crucial to harness technology's potential while maintaining human connection and social values.
                    """,
                "grade": "A" ,
                "feedback": "The essay meets all the requirements and is well-written.",
            }
        }
    }
