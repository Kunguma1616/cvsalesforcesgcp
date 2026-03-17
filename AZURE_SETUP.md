# Azure Storage Setup Guide

## Error: "sp is mandatory. Cannot be empty"

This error means your SAS token is **incomplete or malformed**. The `sp` parameter (Signed Permissions) is required.

## Step-by-Step Setup

### 1. Get Your Storage Account Connection String

1. Go to [Azure Portal](https://portal.azure.com)
2. Find your **Storage Account**
3. Go to **Settings > Access Keys**
4. Copy the **Connection String** (Primary)

Should look like:
```
DefaultEndpointsProtocol=https;AccountName=mystorageaccount;AccountKey=ABC123...;EndpointSuffix=core.windows.net
```

### 2. Create a Container Named "resumes"

1. In your Storage Account, go to **Data Storage > Containers**
2. Click **+ Container**
3. Name it: `resumes`
4. Public access level: **Container**
5. Click **Create**

### 3. Generate a Valid SAS Token

**Option A: Using Azure Portal (Recommended)**

1. Go to your Storage Account
2. **Settings > Shared access signature**
3. Configure:
   - **Allowed services:** Blob
   - **Allowed resource types:** Object
   - **Allowed permissions:** Read, Write, Delete, List, Add, Create, Update
   - **Start time:** Today (or leave empty)
   - **Expiry time:** 1 year from now
   - **Allowed protocols:** HTTPS only

4. Click **Generate SAS and connection string**
5. Copy the **SAS token** (starts with `sv=`)

Should look like:
```
sv=2024-01-01&ss=b&srt=sco&sp=rwdlacupitfx&se=2026-03-13T10:40:00Z&st=2024-03-13T10:40:00Z&spr=https&sig=ABC123...
```

**Key Parameters:**
- `sv` = Signed Version ✓
- `ss` = Signed Services (b=blob) ✓
- `srt` = Signed Resource Types (co=container+object) ✓
- `sp` = Signed Permissions (rwdlacupitfx) ✓
- `se` = Signed Expiry ✓
- `st` = Signed Start ✓
- `spr` = Signed Protocol ✓
- `sig` = Signature ✓

### 4. Update Your .env File

```dotenv
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=mystorageaccount;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net
AZURE_CONTAINER_NAME=resumes
AZURE_SAS_TOKEN=sv=2024-01-01&ss=b&srt=sco&sp=rwdlacupitfx&se=2026-03-13T10:40:00Z&st=2024-03-13T10:40:00Z&spr=https&sig=YOUR_SIGNATURE
```

### 5. Test the Configuration

Restart your backend:
```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then upload a CV - you should see:
- ✓ PDF generated
- ✓ Stored in Azure (with clickable link)
- PDF download button activated

## Troubleshooting

| Error | Solution |
|-------|----------|
| `sp is mandatory` | SAS token missing `sp` parameter - regenerate it |
| `AuthenticationFailed` | SAS token expired or invalid signature - regenerate |
| `Container does not exist` | Create a container named exactly `resumes` |
| `Connection string invalid` | Check `AZURE_STORAGE_CONNECTION_STRING` format |
| `404 Not Found` | PDF upload failed - check permissions in SAS token |

## Verify SAS Token Format

Your SAS token must contain **all** these elements:

```
✓ sv=YYYY-MM-DD          (signed version)
✓ ss=b                   (blob service)
✓ srt=sco                (container + object)
✓ sp=rwdlacupitfx        (permissions)
✓ se=YYYY-MM-DDTHH:MM:Z  (expiry)
✓ st=YYYY-MM-DDTHH:MM:Z  (start time)
✓ spr=https              (protocol)
✓ sig=XXXXX              (signature)
```

If any are missing, regenerate the token in Azure Portal.

## Without Azure Setup (Temporary)

If Azure isn't set up yet, the system will:
- ✓ Still generate PDF
- ✓ Still create Salesforce record (with local file reference)
- ✗ Won't upload to cloud
- Show error notification (but won't crash)

**Next Steps:**
1. Complete Azure setup above
2. Restart backend
3. Try uploading a CV again
