from pydantic import BaseModel, Field
from typing import List
import os
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from app.services.logger import setup_logger
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

load_dotenv(find_dotenv())

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

logger = setup_logger()

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()
    
class EssayGradingAssistant:
    def __init__(self, rubrics, model=None, verbose=False):
        default_config = {
            "model": genai.GenerativeModel(model_name='gemini-1.5-flash', 
                                           system_instruction=read_text_file("prompt/essay-grading-assistant-context.txt"))
        }

        self.rubrics = rubrics
        self.model = model or default_config["model"]   
        self.verbose = verbose
        self.parser = JsonOutputParser(pydantic_object=EssayGradeOutput)

        # Initialize the chat session with Gemini
        self.chat_session = self.model.start_chat()
    
    def validate_output(self, output: dict) -> bool:
        # TODO: Implement output validator if needed
        pass 

    def grade_essay(self, docs: List[Document]):
        # Generate the grade and feedback for the essay
        context = "\n".join([doc.page_content for doc in docs])
        
        response = self.chat_session.send_message(f"""
            Here are the rubrics for grading the essay: \n
            {self.rubrics} \n
            Below is the essay to be graded: \n
            {context} \n
            Output the grades and feedback in the following format: \n
            {self.parser.get_format_instructions()} \n
            Based on the rubrics and the conversation history, provide personalized grades and feedback on the student essay. If you need more information, ask the educator for clarification.\n
            """)
        
        output = self.parser.parse(response.text) 

        # Validate the output here

        return output

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
