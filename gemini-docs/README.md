# API-AISAC: PDF Document Analysis with Gemini AI

A FastAPI-based REST API that extracts and analyzes text from PDF documents stored on an FTP server using Google's Gemini AI. The API allows users to ask questions about the content of PDF documents and receive AI-powered responses.

## Features

- 📄 PDF text extraction using PyMuPDF
- 🤖 Text analysis using Google's Gemini AI
- 📡 FTP integration for PDF document retrieval
- 🔐 Environment-based configuration
- 🚀 Deployed on Vercel

## API Endpoints

### POST `/analyze_document/{pdf_filename}`

Analyzes a PDF document with a specific prompt.

**Parameters:**
- `pdf_filename`: Name of the PDF file on the FTP server (path parameter)

**Request Body:**
```json
{
    "prompt": "Your question about the PDF content"
}
```

**Response:**
```json
{
    "filename": "example.pdf",
    "prompt_received": "Your question about the PDF content",
    "analysis_result": "AI-generated analysis of the PDF content"
}
```

## Environment Variables

The following environment variables are required:

- `GEMINI_API_KEY`: Google Gemini AI API key
- `FTP_HOST`: FTP server host address
- `FTP_USER`: FTP server username
- `FTP_PASS`: FTP server password

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Anthonyzok521/api-aisac.git
cd api-aisac
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
GEMINI_API_KEY=your_gemini_api_key
FTP_HOST=your_ftp_host
FTP_USER=your_ftp_username
FTP_PASS=your_ftp_password
```

## Development

Run the development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Deployment

The project is configured for deployment on Vercel. The production API is available at:
https://api-aisac-9nkop96s0-anthonyzok521s-projects.vercel.app

### Deploying to Vercel

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy:
```bash
vercel
```

3. Add environment variables:
```bash
vercel env add GEMINI_API_KEY
vercel env add FTP_HOST
vercel env add FTP_USER
vercel env add FTP_PASS
```

## Dependencies

- FastAPI
- Uvicorn
- PyMuPDF
- Google Generative AI
- python-dotenv
- python-multipart
- Pydantic

## License

MIT License - feel free to use this project for your own purposes.

## Author

Anthony Chávez <anthonyzok521@gmail.com>

