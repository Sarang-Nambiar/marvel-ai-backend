from app.services.logger import setup_logger
from app.utils.document_loaders import get_docs
from app.assistants.classroom_support.essay_grading_assistant.assistant import EssayGradingAssistant

logger = setup_logger()

def executor(
            rubrics: str,
            file_url: str,
            file_type: str,
            verbose=False
            ):
    
    try:    
        if not file_type or not file_url:
            raise ("File URL and File Type are required")

        if verbose: logger.info(f"Generating docs from {file_type}")

        docs = get_docs(file_url, file_type, verbose=verbose)

        # TODO: Create and return the grade with feedback
        output = EssayGradingAssistant(rubrics=rubrics, verbose=verbose).grade_essay(docs)

        logger.info(f"Essay Grading Assistant executed successfully")

    except Exception as e:
        logger.error(f"Error in executing Essay Grading Assistant: {e}")
        raise ValueError(f"Error in executing Essay Grading Assistant: {e}")
    
    return output