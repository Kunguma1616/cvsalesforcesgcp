# Resume Analyzer - Setup Instructions

## System Architecture

- **Frontend**: React + Vite on port 3000
- **Backend FastAPI**: Port 8000 (for frontend API calls)
- **Backend Streamlit**: Port 8501 (standalone Streamlit UI - optional)

## Prerequisites

1. **Python 3.10+** installed
2. **Node.js 18+** installed (for frontend)
3. **Groq API Key** - Get from [console.groq.com](https://console.groq.com)
4. **Azure Storage Account** (for PDF storage)
5. **Salesforce Account** (for record creation)

## Environment Setup

### 1. Backend Configuration

Create `.env` file in the `backend/` directory:

```bash
cp backend/.env.example backend/.env
```

Then edit `backend/.env` with:

#### GROQ API
```
GROQ_API_KEY=xxxxxxxxxxxx (from console.groq.com)
GROQ_MODEL=llama-3.3-70b-versatile
```

#### Azure Storage
1. Go to [Azure Portal](https://portal.azure.com)
2. Create a Storage Account
3. Get Connection String from Settings > Access Keys
4. Create a container named `resumes`
5. Generate SAS URL from Shared access signature

```
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_CONTAINER_NAME=resumes
AZURE_SAS_TOKEN=sv=2024-01-01&ss=bfqt&...
```

#### Salesforce
1. Go to [Salesforce Setup](https://login.salesforce.com)
2. Get your Security Token from Settings > Personal Information
3. Create API Integration User if needed

```
SF_USERNAME=your_email@example.com
SF_PASSWORD=your_password
SF_SECURITY_TOKEN=your_security_token
SF_DOMAIN=login
```

### 2. Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

## Running the Application

### Option 1: FastAPI Backend + React Frontend (Recommended)

**Terminal 1 - Start Backend API:**
```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```

Access at: **http://localhost:3000**

### Option 2: Streamlit UI (Standalone)

**Terminal 1 - Start Streamlit App:**
```bash
cd backend
streamlit run app.py --server.port 8501
```

Access at: **http://localhost:8501**

## Workflow

### Using React Frontend (http://localhost:3000)

1. Navigate to `/upload` page
2. Fill in candidate details:
   - Full Name
   - Email Address
   - Trade / Profession
   - Job Description
3. Upload PDF Resume
4. Click **Upload CV for Analysis**
5. View results on analysis page
6. System automatically:
   - Analyzes resume with AI
   - Calculates ATS score
   - Generates PDF report
   - Uploads to Azure Storage
   - Creates Salesforce record

### Using Streamlit UI (http://localhost:8501)

1. Fill in the form with candidate details
2. Upload PDF resume
3. Click **Analyze Resume**
4. Click **Generate & Store Report**
5. System will:
   - Generate PDF report
   - Upload to Azure
   - Create Salesforce record
6. Download PDF if needed

## Expected Output

### Scores
- **ATS Keyword Match**: 0-100% based on resume-job description keyword overlap
- **AI Evaluation Score**: 0-100% based on 7-10 evaluation criteria

### Analysis Includes
- Job classification (category + subcategory)
- 7-10 criteria with scores and detailed explanations
- 6-10 strengths and weaknesses
- Requirements met/missing
- Skills identified, relevant, and missing
- Overall HR assessment (8-12 sentences)

### Storage
- **PDF Report**: Uploaded to Azure Storage with public SAS URL
- **Salesforce Record**: Engineer_Application__c record created with:
  - First/Last Name
  - Email
  - Resume URL (Azure link)
  - Primary Trade

## Troubleshooting

### 500 Error from Frontend
- Ensure FastAPI is running on port 8000
- Check GROQ_API_KEY is set and valid
- Verify CORS settings in main.py

### Azure Upload Fails
- Check `AZURE_STORAGE_CONNECTION_STRING` is correct
- Verify `AZURE_CONTAINER_NAME` exists
- Ensure `AZURE_SAS_TOKEN` is valid and not expired

### Salesforce Record Not Created
- Verify credentials in .env
- Check Salesforce Engineer_Application__c object exists
- Ensure user has API integration permissions
- Check field names match: First_Name__c, Last_Name__c, Email_Address__c, Your_CV__c, Primary_Trade__c

### PDF Generation Issues
- Ensure reportlab is installed
- Check logo file path is correct
- Verify matplotlib is properly configured

## API Endpoints

### FastAPI (Port 8000)

```
GET /health
- Returns: {"status": "ok", "model": "llama-3.3-70b-versatile"}

POST /analyze
- Form Data:
  - name: string
  - email: string
  - trade: string
  - job_description: string
  - resume: PDF file
- Returns: JSON with analysis, ATS score, AI score
```

## Notes

- Trade list updated to 32 specific trade options
- All trades properly mapped to Salesforce API values
- Frontend proxy configured to redirect /api/* to http://localhost:8000
- PDF reports include scores, charts, and detailed analysis
- Session state prevents data loss during analysis
- All personal data logged to Salesforce and Azure for compliance

## Security

- Use environment variables for all credentials
- Never commit .env file (included in .gitignore)
- SAS tokens should have expiration dates
- Salesforce password stored securely in encrypted format
- PDF links include SAS tokens for access control
