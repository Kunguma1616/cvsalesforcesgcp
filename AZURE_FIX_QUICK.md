# Fixing Azure Storage Upload Issue

## The Problem You're Seeing:

```xml
<Error>
<Code>AuthenticationFailed</Code>
<Message>Server failed to authenticate the request...</Message>
<AuthenticationErrorDetail>sp is mandatory. Cannot be empty</AuthenticationErrorDetail>
</Error>
```

## What This Means:

Your SAS token in `.env` is **invalid or incomplete**. The `sp` parameter (Signed Permissions) is required but missing.

## Quick Fix (3 Steps):

### Step 1: Generate a New SAS Token in Azure Portal

1. Open [Azure Portal](https://portal.azure.com)
2. Go to your **Storage Account** → **Settings** → **Shared access signature**
3. Configuration:
   ```
   Allowed services:     ✓ Blob
   Resource types:       ✓ Container, Object
   Permissions:          ✓ Read, Write, Delete, List, Add, Create, Update
   Start time:           (leave empty or today)
   Expiry:               1 year from today
   Allowed protocols:    HTTPS only
   ```

4. Click **Generate SAS and connection string**
5. **Copy the SAS token** (the part that starts with `sv=`)

### Step 2: Update `.env` File

Open `backend/.env` and make sure it has:

```dotenv
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net
AZURE_CONTAINER_NAME=resumes
AZURE_SAS_TOKEN=sv=2024-01-01&ss=b&srt=sco&sp=rwdlacupitfx&se=2027-03-13T10:40:00Z&st=2024-03-13T10:40:00Z&spr=https&sig=YOUR_SIGNATURE
```

**Important:** Your SAS token MUST contain:
- `sv=` (signed version)
- `ss=b` (blob service)  
- `srt=sco` (container + object)
- `sp=rwdlacupitfx` (PERMISSIONS) ← **This is what was missing!**
- `se=` (expiry date)
- `sig=` (signature)

### Step 3: Restart Backend

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Test It

1. Go to http://localhost:3000/upload
2. Upload a CV
3. You should now see:
   - ✓ PDF Report Generated (Download button)
   - ✓ Stored in Azure (View link)
   - ✓ Record ID from Salesforce

## If Still Getting Error:

Check that:
1. ✓ Container named `resumes` exists in Azure
2. ✓ SAS token contains `sp=rwdlacupitfx` (the signed permissions)
3. ✓ SAS token has `se=FUTURE_DATE` (not expired)
4. ✓ Connection string doesn't have typos
5. ✓ Restart backend after updating .env

## System Now Shows:

- **✓ Green box** = PDF uploaded successfully
- **⚠️ Yellow box** = Azure upload failed (shows exact error)
- **✓ Green box** = Record saved to Salesforce
- **❌ Red box** = Salesforce error (shows why)

The system will **continue working even if Azure fails** - but PDF won't be stored in the cloud.

## See Full Guide

Check `AZURE_SETUP.md` for complete instructions with screenshots.
