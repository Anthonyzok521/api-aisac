import os
import ftplib
import tempfile
import fitz  # PyMuPDF
import uvicorn
import logging
import google.generativeai as genai
import google.api_core.exceptions
# Added BackgroundTasks
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from io import BytesIO
from datetime import datetime
import shutil
import asyncio # For running async background task properly

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
FTP_HOST = os.getenv("FTP_HOST")
FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- AISAC Configuration ---
SYSTEM_INSTRUCTION = """Tu nombre es AISAC (Artificial Intelligence Service for Academic Consults, y en español significa, Servicio de Inteligencia Artificial para Consultas Académicas). Eres un asistente de una universidad venezolana del estado Guárico, San Juan de los Morros. La universidad es la UNERG (Universidad Nacional Experimental Rómulo Gallegos).

Tu propósito es ofrecer información a los estudiantes y profesores con respecto a solo preguntas académicas, basándote principalmente en el CONOCIMIENTO COMBINADO DE LOS DOCUMENTOS DE ENTRENAMIENTO CARGADOS y tu conocimiento general. Recuerda que antes que nada debes de ser amable y siempre hablar en primera persona.

Si te hacen preguntas que no tienen nada que ver con la educación o tema académico, responde amablemente que tu función se limita a consultas académicas de la UNERG.

Si te preguntan quién es tu creador, respondes: Mi creador es Anthony Carrillo, un ingeniero de informática. Es una gran persona y bastante reconocido en el área de ingeniería de sistemas.

Puedes recibir información desde documentos PDF proporcionados y tu base de conocimiento interna (derivada de los documentos de entrenamiento). Indica claramente si tu respuesta se basa en esta base de conocimiento interna o si es conocimiento general. Si no encuentras la información relevante, indícalo también.

Aquí tienes algunos enlaces útiles que puedes compartir si es relevante:
Facebook Oficial UNERG: https://www.facebook.com/oficialunerg1977/
DACE Inscripciones: https://cde.unerg.edu.ve/
DACE Admisión (Info Carreras): https://dace.unerg.edu.ve/
Wikipedia UNERG: https://es.wikipedia.org/wiki/Universidad_Nacional_Experimental_R%C3%B3mulo_Gallegos
Coseca (Servicio Comunitario): https://ais.coseca.top
"""  # Updated SYSTEM_INSTRUCTION
TRAIN_FILE_FTP_PATH = "train.md"
TRAIN_DOCS_FTP_DIR = "train-docs"

# --- Global variable for combined training text ---
combined_training_text: str = "Training data not loaded yet."
# Lock for safely updating the global variable from background tasks
training_data_lock = asyncio.Lock()

# --- Configure Gemini ---
try:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY environment variable not found.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Error configuring Gemini API: {e}", exc_info=True)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AISAC Document Analyzer API",
    description="Uses Gemini for analysis, supports dynamic training document loading.",
    version="1.3.0"  # Incremented version
)

# --- Pydantic Models ---
class ExtractionRequest(BaseModel):
    prompt: str

class AnalysisResponse(BaseModel):
    filename: str
    prompt_received: str
    analysis_result: str

class TrainingResponse(BaseModel):
    message: str
    filename: str
    prompt_added: str
    summary_added: str # Changed from text_extracted_preview

class AssistantRequest(BaseModel):
    question: str

class AssistantResponse(BaseModel):
    response: str

# --- FTP Helper Functions ---
# (connect_ftp, ensure_ftp_dir, read_ftp_file, append_to_ftp_file are mostly unchanged)
# Minor improvement: ensure_ftp_dir returns to root if original dir fails
def connect_ftp() -> ftplib.FTP:
    """Connects and logs in to the FTP server."""
    if not all([FTP_HOST, FTP_USER, FTP_PASS]):
        logger.error("FTP server configuration is missing.")
        raise HTTPException(status_code=500, detail="FTP server configuration is missing.")
    try:
        # Increased timeout for potentially slow connections
        ftp = ftplib.FTP(timeout=30)
        ftp.connect(FTP_HOST)
        ftp.login(user=FTP_USER, passwd=FTP_PASS)
        # Enable UTF-8 support if server supports it
        try:
            ftp.sendcmd('OPTS UTF8 ON')
            logger.info("Enabled UTF-8 mode for FTP connection.")
        except ftplib.all_errors:
            logger.warning("FTP server may not support UTF-8.")
        logger.info(f"FTP Connected and logged in as {FTP_USER}")
        return ftp
    except ftplib.all_errors as e:
        logger.error(f"FTP connection/login failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"FTP connection failed: {e}")
    except Exception as e: # Catch other errors like timeout
        logger.error(f"FTP connection failed (General Error): {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"FTP connection failed: {e}")


def ensure_ftp_dir(ftp: ftplib.FTP, dir_path: str):
    """Checks if a directory exists on FTP, creates it if not."""
    current_dir = ftp.pwd()
    try:
        ftp.cwd(dir_path)
        logger.info(f"FTP directory '{dir_path}' already exists.")
    except ftplib.error_perm as e:
        if "550" in str(e):
            try:
                logger.info(f"Attempting to create FTP directory: {dir_path}")
                ftp.mkd(dir_path)
                logger.info(f"Created FTP directory: {dir_path}")
            except ftplib.all_errors as mkd_e:
                logger.error(f"Failed to create FTP directory '{dir_path}': {mkd_e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to create FTP directory '{dir_path}': {mkd_e}")
        else:
            logger.error(f"FTP permission error checking/creating directory '{dir_path}': {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"FTP permission error for directory '{dir_path}': {e}")
    except ftplib.all_errors as e:
         logger.error(f"FTP error checking/creating directory '{dir_path}': {e}", exc_info=True)
         raise HTTPException(status_code=503, detail=f"FTP error for directory '{dir_path}': {e}")
    finally:
        try:
            ftp.cwd(current_dir)
        except ftplib.all_errors:
            logger.warning(f"Could not return to original FTP directory '{current_dir}', attempting to go to root '/'")
            try:
                ftp.cwd('/') # Fallback to root
            except ftplib.all_errors:
                 logger.error("Failed to navigate back to a known FTP directory.")


def read_ftp_file(ftp: ftplib.FTP, remote_path: str) -> str:
    """Reads the content of a text file from FTP."""
    content = ""
    try:
        r = BytesIO()
        ftp.retrbinary(f'RETR {remote_path}', r.write)
        r.seek(0)
        content = r.read().decode('utf-8', errors='ignore')
        logger.info(f"Successfully read content from FTP file: {remote_path}")
    except ftplib.error_perm as e:
        if "550" in str(e):
            logger.warning(f"FTP file not found: {remote_path}. Returning empty content.")
            content = ""
        else:
            logger.error(f"FTP permission error reading file '{remote_path}': {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"FTP permission error reading file '{remote_path}': {e}")
    except ftplib.all_errors as e:
        logger.error(f"FTP error reading file '{remote_path}': {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"FTP error reading file '{remote_path}': {e}")
    except Exception as e:
        logger.error(f"Error processing content from FTP file '{remote_path}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing content from FTP file '{remote_path}': {e}")
    return content

def append_to_ftp_file(ftp: ftplib.FTP, remote_path: str, text_to_append: str):
    """Appends text to a file on the FTP server."""
    try:
        existing_content = read_ftp_file(ftp, remote_path)
        separator = "\n\n" if existing_content.strip() else ""
        new_content = existing_content.strip() + separator + text_to_append.strip() + "\n"

        with BytesIO(new_content.encode('utf-8')) as bio:
            ftp.storbinary(f'STOR {remote_path}', bio)
        logger.info(f"Successfully appended text to FTP file: {remote_path}")

    except ftplib.all_errors as e:
        logger.error(f"FTP error appending to file '{remote_path}': {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"FTP error appending to file '{remote_path}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error appending to FTP file '{remote_path}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error appending to FTP file '{remote_path}': {e}")

# upload_file_obj_to_ftp remains the same
async def upload_file_obj_to_ftp(ftp: ftplib.FTP, file_obj: UploadFile, remote_path: str):
    """Uploads a file object (like FastAPI's UploadFile) to FTP."""
    try:
        file_obj.file.seek(0)
        ftp.storbinary(f'STOR {remote_path}', file_obj.file)
        logger.info(f"Successfully uploaded file to FTP: {remote_path}")
    except ftplib.all_errors as e:
        logger.error(f"FTP error uploading file to '{remote_path}': {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"FTP error uploading file to '{remote_path}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error uploading file to FTP '{remote_path}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error uploading file to FTP '{remote_path}': {e}")


# --- Text Extraction and Summarization ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file."""
    # (Implementation remains the same as previous version)
    doc = None
    extracted_text = ""
    try:
        logger.info(f"Opening PDF file for text extraction: {pdf_path}")
        doc = fitz.open(pdf_path)
        extracted_text = "\n".join([page.get_text() for page in doc])
        logger.info(f"Successfully extracted text from {len(doc)} pages.")
    except fitz.fitz.FileNotFoundError:
        logger.error(f"PDF file not found at path: {pdf_path}")
        raise FileNotFoundError(f"Temporary PDF file not found: {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to process PDF file '{pdf_path}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to process PDF file: {e}")
    finally:
        if doc:
            try:
                doc.close()
                logger.info(f"Closed PDF document: {pdf_path}")
            except Exception as e:
                 logger.warning(f"Error closing PDF document {pdf_path}: {e}")
    return extracted_text

# --- NEW Summarization Function ---
def summarize_text_with_gemini(text_content: str, model_name: str = "gemini-2.0-flash") -> str:
    """Uses Gemini to summarize the provided text."""
    if not GEMINI_API_KEY:
        logger.error("Gemini API key not configured. Cannot summarize text.")
        # Return original text or raise error? Returning truncated original for now.
        return text_content[:500] + "... (Summarization failed: API key missing)"

    try:
        logger.info(f"Initializing Gemini model for summarization: {model_name}")
        # Use a model optimized for speed if possible, like Flash
        model = genai.GenerativeModel(model_name)

        # Limit input text size for summarization if necessary (Flash has large context)
        max_summary_input = 900000
        truncated_content = text_content[:max_summary_input]
        if len(text_content) > max_summary_input:
             logger.warning(f"Text content truncated to {max_summary_input} chars for summarization.")

        prompt = f"Por favor, resume el siguiente texto de manera concisa y clara, capturando las ideas principales:\n\n---\n{truncated_content}\n---"

        logger.info("Sending summarization request to Gemini API...")
        response = model.generate_content(prompt)

        # Basic check (can reuse parts of generate_gemini_response error handling)
        try:
            summary = response.text
            if not summary:
                 raise ValueError("Empty summary received")
            logger.info("Successfully received summary from Gemini.")
            return summary.strip()
        except (ValueError, AttributeError) as e:
             logger.error(f"Failed to get valid summary from Gemini response: {e}. Response: {response}")
             # Fallback or re-raise
             return text_content[:500] + "... (Summarization failed: Invalid response)"

    except Exception as e:
        logger.error(f"Error calling Gemini API for summarization: {e}", exc_info=True)
        # Fallback or re-raise
        return text_content[:500] + f"... (Summarization failed: API error {e})"


# --- NEW Training Data Loading Function ---
async def load_all_training_data():
    """
    Connects to FTP, lists files in TRAIN_DOCS_FTP_DIR, downloads PDFs,
    extracts text, and updates the global combined_training_text.
    Designed to be run in the background.
    """
    global combined_training_text
    logger.info("Starting background task: Load all training data from FTP...")
    ftp = None
    all_texts = []
    temp_files_to_delete = []

    try:
        ftp = connect_ftp()
        try:
            ftp.cwd(TRAIN_DOCS_FTP_DIR)
            logger.info(f"Navigated to FTP directory: {TRAIN_DOCS_FTP_DIR}")
        except ftplib.error_perm as e:
             if "550" in str(e):
                 logger.warning(f"Training directory '{TRAIN_DOCS_FTP_DIR}' not found on FTP. No data loaded.")
                 async with training_data_lock:
                      combined_training_text = "No training documents found."
                 return # Nothing to load
             else:
                 raise # Re-raise other permission errors

        filenames = ftp.nlst()
        pdf_files = [f for f in filenames if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files in {TRAIN_DOCS_FTP_DIR}.")

        for filename in pdf_files:
            temp_pdf_path = None
            try:
                # Create a temporary file for download
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf_path = temp_pdf.name
                temp_files_to_delete.append(temp_pdf_path) # Add to cleanup list

                # Download
                logger.info(f"Downloading training file: {filename}...")
                full_remote_path = f"{TRAIN_DOCS_FTP_DIR}/{filename}" # Ensure full path
                with open(temp_pdf_path, 'wb') as fp:
                    # Use cwd or full path depending on how connect_ftp leaves the state
                    # Assuming we are in TRAIN_DOCS_FTP_DIR now
                    ftp.retrbinary(f'RETR {filename}', fp.write)
                logger.info(f"Downloaded to temporary file: {temp_pdf_path}")

                # Extract text
                extracted = extract_text_from_pdf(temp_pdf_path)
                if extracted:
                    # Add separator and filename for context
                    all_texts.append(f"\n\n--- START OF DOCUMENT: {filename} ---\n\n")
                    all_texts.append(extracted)
                    all_texts.append(f"\n\n--- END OF DOCUMENT: {filename} ---\n\n")
                    logger.info(f"Extracted text from {filename}.")
                else:
                    logger.warning(f"No text extracted from {filename}.")

            except ftplib.all_errors as ftp_dl_err:
                logger.error(f"FTP error downloading/processing {filename}: {ftp_dl_err}", exc_info=True)
                # Continue to next file
            except (FileNotFoundError, RuntimeError, Exception) as proc_err:
                logger.error(f"Error processing local file for {filename}: {proc_err}", exc_info=True)
                # Continue to next file
            finally:
                 # Clean up individual temp file immediately after use? Or at the end?
                 # Let's keep track and delete at the end for simplicity here.
                 pass

        # Update global variable safely
        async with training_data_lock:
            if all_texts:
                combined_training_text = "".join(all_texts)
                logger.info(f"Successfully loaded and combined text from {len(pdf_files)} training documents. Total length: {len(combined_training_text)}")
            else:
                combined_training_text = "No text could be extracted from training documents."
                logger.info("No text extracted from any training documents.")

    except HTTPException as http_exc:
        logger.error(f"HTTPException during training data load: {http_exc.detail}")
        async with training_data_lock:
             combined_training_text = f"Error loading training data: {http_exc.detail}"
    except Exception as e:
        logger.error(f"Unexpected error loading training data: {e}", exc_info=True)
        async with training_data_lock:
             combined_training_text = f"Unexpected error loading training data: {e}"
    finally:
        if ftp:
            try:
                ftp.quit()
                logger.info("FTP connection closed after loading training data.")
            except ftplib.all_errors as e:
                logger.warning(f"Error closing FTP connection after loading data: {e}")
        # Clean up all temporary files created
        for f_path in temp_files_to_delete:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    logger.info(f"Cleaned up temporary file: {f_path}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary file {f_path}: {e}")
        logger.info("Finished background task: Load all training data.")


# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_load_training_data():
    """Loads training data when the application starts."""
    logger.info("Application startup: Triggering initial load of training data.")
    # Run in background to avoid blocking startup if FTP is slow,
    # although it might mean the first few requests don't have data.
    # Alternatively, run synchronously if data MUST be available at start.
    # Let's run synchronously for simplicity at startup.
    await load_all_training_data()


# --- MODIFIED Gemini Interaction Logic ---
def generate_gemini_response(
    user_query: str,
    system_instruction: str = SYSTEM_INSTRUCTION,
    document_context: str | None = None,
    # Changed parameter name and source
    in_memory_training_data: str | None = None,
    model_name: str = "gemini-2.0-flash"
) -> str:
    """
    Generates a response from Gemini based on a user query, system instruction,
    optional specific document context, and in-memory combined training data.
    """
    # (Error handling for API key remains the same)
    if not GEMINI_API_KEY:
        logger.error("Gemini API key is not configured. Cannot perform analysis.")
        raise HTTPException(status_code=503, detail="Gemini service is not configured on the server.")

    try:
        logger.info(f"Initializing Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)

        prompt_parts = [system_instruction]
        # Use a large context limit, Gemini 1.5 can handle a lot
        context_limit = 950000 # Slightly less than 1M for safety

        # Include combined training data from memory if available
        if in_memory_training_data and in_memory_training_data != "Training data not loaded yet." and "Error loading" not in in_memory_training_data:
            truncated_training_data = in_memory_training_data[:context_limit]
            if len(in_memory_training_data) > context_limit:
                logger.warning(f"In-memory training data truncated to {context_limit} characters.")
            prompt_parts.append("\n---\n**Knowledge Base (from Training Documents):**\n" + truncated_training_data + "\n---")
            context_limit -= len(truncated_training_data) # Adjust remaining limit
        else:
             logger.warning("No valid in-memory training data available for this request.")


        # Include specific document context if available (and limit allows)
        if document_context and context_limit > 100:
            truncated_doc_context = document_context[:context_limit]
            if len(document_context) > context_limit:
                logger.warning(f"Specific document context truncated to {context_limit} characters.")
            prompt_parts.append("\n---\n**Specific Document Content (for this query):**\n" + truncated_doc_context + "\n---")
            prompt_parts.append(f"\n**User Request (related to specific document):**\n{user_query}")
            prompt_parts.append("\n---\nBased *primarily* on the 'Specific Document Content' provided above, but also considering the 'Knowledge Base', and adhering to your persona (AISAC), please fulfill the user request. If the information needed is not in the specific document, state that clearly but check the knowledge base if relevant.")
        else:
            # General query using knowledge base
            prompt_parts.append(f"\n**User Request:**\n{user_query}")
            if in_memory_training_data:
                 prompt_parts.append("\n---\nAdhering to your persona (AISAC), please answer the user's request using your general knowledge and the provided 'Knowledge Base (from Training Documents)'. Mention if the answer comes primarily from the knowledge base or general knowledge.")
            else:
                 prompt_parts.append("\n---\nAdhering to your persona (AISAC), please answer the user's request using your general knowledge. Be helpful and academic-focused.")


        full_prompt = "\n".join(prompt_parts)

        logger.info("Sending request to Gemini API...")
        # logger.debug(f"Full prompt length: {len(full_prompt)}")
        response = model.generate_content(full_prompt)

        # --- Response processing (remains the same) ---
        try:
             analysis_text = response.text
        except ValueError:
             logger.warning(f"Gemini response might be empty or blocked. Raw response: {response}")
             block_reason = None; safety_ratings = None
             try:
                 if response.prompt_feedback and response.prompt_feedback.block_reason: block_reason = response.prompt_feedback.block_reason
                 if response.candidates and response.candidates[0].safety_ratings: safety_ratings = response.candidates[0].safety_ratings
             except (AttributeError, IndexError): logger.warning("Could not access detailed feedback/safety ratings.")
             if block_reason: raise HTTPException(status_code=400, detail=f"Request blocked by content safety filter: {block_reason}")
             elif safety_ratings: raise HTTPException(status_code=500, detail="Gemini response flagged due to safety concerns.")
             else: raise HTTPException(status_code=500, detail="Gemini returned an unexpected response structure.")
        if not analysis_text: raise HTTPException(status_code=500, detail="Gemini returned an empty response text.")

        logger.info("Received response from Gemini API.")
        return analysis_text.strip()

    # --- Exception handling (remains the same) ---
    except google.api_core.exceptions.NotFound as e:
        logger.error(f"Gemini model '{model_name}' not found or not supported: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"The specified AI model ('{model_name}') was not found or is not supported.")
    except google.api_core.exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error calling Gemini: {e}", exc_info=True)
        status_code = 503
        if hasattr(e, 'code'):
             if e.code == 429: status_code = 429
             elif e.code == 400: status_code = 400 # Could be prompt too long
        raise HTTPException(status_code=status_code, detail=f"Failed to get analysis from AI service (API Error): {e}")
    except Exception as e:
        logger.error(f"Unexpected error calling Gemini API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while communicating with the AI service: {e}")


# --- API Endpoints ---

@app.post("/analyze_document/{pdf_filename}", response_model=AnalysisResponse)
async def analyze_pdf_document(pdf_filename: str, request_data: ExtractionRequest):
    """
    Downloads PDF, extracts text, sends to Gemini for analysis based on the document
    AND the in-memory knowledge base.
    """
    logger.info(f"Received analysis request for file: {pdf_filename} with prompt: '{request_data.prompt}'")

    if not pdf_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Filename must end with .pdf")

    temp_pdf_path = None
    try:
        temp_pdf_path = download_pdf_from_ftp(pdf_filename)
        extracted_text = extract_text_from_pdf(temp_pdf_path)

        # Access the global training data safely (read-only access here is fine without lock)
        current_training_data = combined_training_text

        ai_analysis_result = generate_gemini_response(
            user_query=request_data.prompt,
            document_context=extracted_text,
            in_memory_training_data=current_training_data # Pass the loaded training data
        )
        logger.info(f"Successfully processed and analyzed file: {pdf_filename}")

        return AnalysisResponse(
            filename=pdf_filename,
            prompt_received=request_data.prompt,
            analysis_result=ai_analysis_result
        )
    # ... (exception handling and finally block remain the same) ...
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred processing {pdf_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try: os.remove(temp_pdf_path); logger.info(f"Removed temporary PDF file: {temp_pdf_path}")
            except OSError as e: logger.warning(f"Could not remove temporary file {temp_pdf_path}: {e}")


# --- MODIFIED Training Endpoint ---
@app.post("/train-aisac", response_model=TrainingResponse, status_code=202) # Changed to 202 Accepted for background task
async def train_aisac(
    background_tasks: BackgroundTasks, # Inject background tasks
    prompt: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Accepts PDF and prompt. Saves PDF to FTP, generates summary, appends
    prompt and summary to train.md, and triggers a background task
    to reload all training data into memory.
    """
    logger.info(f"Received training request. File: {file.filename}, Prompt: '{prompt[:100]}...'")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type or filename. Only PDF files are allowed.")

    ftp = None
    remote_pdf_path = None
    temp_pdf_path = None
    extracted_text = ""
    document_summary = ""

    try:
        # 1. Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf_path = temp_pdf.name
            shutil.copyfileobj(file.file, temp_pdf)
            logger.info(f"Saved uploaded file temporarily to: {temp_pdf_path}")
        await file.close() # Close the UploadFile stream

        # 2. Extract text
        try:
            extracted_text = extract_text_from_pdf(temp_pdf_path)
            logger.info(f"Extracted text from temporary file: {file.filename}")
        except (FileNotFoundError, RuntimeError) as extract_err:
             logger.error(f"Failed to extract text during training: {extract_err}")
             raise HTTPException(status_code=500, detail=f"Failed to extract text from uploaded PDF: {extract_err}")

        # --- NEW: 3. Generate Summary ---
        if extracted_text:
             document_summary = summarize_text_with_gemini(extracted_text)
             logger.info(f"Generated summary for {file.filename}")
        else:
             document_summary = "(No text extracted from PDF to summarize)"
             logger.warning(f"Skipping summary generation for {file.filename} due to no extracted text.")


        # 4. Connect to FTP
        ftp = connect_ftp()

        # 5. Ensure training directory exists
        ensure_ftp_dir(ftp, TRAIN_DOCS_FTP_DIR)

        # 6. Upload the original PDF document
        remote_pdf_path = f"{TRAIN_DOCS_FTP_DIR}/{file.filename}"
        try:
            with open(temp_pdf_path, "rb") as temp_pdf_read:
                ftp.storbinary(f'STOR {remote_pdf_path}', temp_pdf_read)
            logger.info(f"Successfully uploaded original PDF to FTP: {remote_pdf_path}")
        except ftplib.all_errors as e:
            logger.error(f"FTP error uploading original PDF to '{remote_pdf_path}': {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"FTP error uploading original PDF to '{remote_pdf_path}': {e}")
        except Exception as e:
             logger.error(f"Unexpected error uploading original PDF to FTP '{remote_pdf_path}': {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Unexpected error uploading original PDF to FTP '{remote_pdf_path}': {e}")

        # 7. Append the prompt and SUMMARY to the training file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # No more truncation needed for summary
        entry_to_add = (
            f"## Training Entry ({timestamp})\n\n"
            f"**Associated File:** `{file.filename}`\n\n"
            f"**Instruction/Prompt:**\n```\n{prompt}\n```\n\n"
            # Changed to add summary
            f"**Generated Summary:**\n```text\n{document_summary}\n```"
        )
        append_to_ftp_file(ftp, TRAIN_FILE_FTP_PATH, entry_to_add)

        # --- NEW: 8. Add background task to reload data ---
        background_tasks.add_task(load_all_training_data)
        logger.info(f"Added background task to reload training data after processing {file.filename}")

        logger.info(f"Successfully processed training request for {file.filename}")

        return TrainingResponse(
            # Updated message and field name
            message="AISAC training log updated with summary. Knowledge base refresh initiated in background.",
            filename=file.filename,
            prompt_added=prompt,
            summary_added=document_summary # Changed field name
        )

    # ... (exception handling remains similar) ...
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred during training for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during training: {e}")
    finally:
        # Close FTP connection if open
        if ftp:
            try: ftp.quit(); logger.info("FTP connection closed after training task.")
            except ftplib.all_errors as e: logger.warning(f"Error closing FTP connection after training: {e}")
        # Clean up the temporary PDF file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try: os.remove(temp_pdf_path); logger.info(f"Removed temporary training PDF file: {temp_pdf_path}")
            except OSError as e: logger.warning(f"Could not remove temporary training file {temp_pdf_path}: {e}")
        # Ensure UploadFile is closed (already closed after copyfileobj, but good practice)
        if file and hasattr(file, 'file') and not file.file.closed:
             await file.close()
             logger.info(f"Ensured UploadFile {file.filename} is closed in finally block.")


# --- MODIFIED Assistant Endpoint ---
@app.post("/assistant", response_model=AssistantResponse)
async def get_assistant_answer(request: AssistantRequest):
    """
    Provides answers using AISAC's persona, general knowledge, and the
    in-memory combined training data.
    """
    logger.info(f"Received assistant request: '{request.question[:100]}...'")

    # No FTP access needed here anymore

    try:
        # Access the global training data safely
        # Read-only access is generally safe without a lock here,
        # but using it ensures we get a consistent state if a background
        # update is somehow in progress (though unlikely with GIL).
        async with training_data_lock:
             current_training_data = combined_training_text

        # Call Gemini with the user query and the in-memory training data
        ai_response = generate_gemini_response(
            user_query=request.question,
            document_context=None,
            in_memory_training_data=current_training_data # Pass the global data
        )
        logger.info("Successfully generated assistant response.")
        return AssistantResponse(response=ai_response)

    # ... (exception handling remains the same) ...
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in /assistant endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred while getting assistant response.")


# --- Run the application ---
if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    # Ensure reload=False in production
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

