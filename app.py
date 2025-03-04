from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header
import subprocess
import json
import os
from pathlib import Path
import glob
from typing import Optional
import sys
python_executable = sys.executable

app = FastAPI()

@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    hf_token: str = Header(..., description="Hugging Face API token"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers")
):
    try:
        # 1. Save the uploaded file temporarily
        temp_audio_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        with open(temp_audio_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 2. Prepare command for transcribe.py with correct argument format
        command = [python_executable, "transcribe.py", "--audio_file", temp_audio_path, "--hf_token", hf_token]
        
        # Add optional parameters if provided
        if min_speakers is not None:
            command.extend(["--min_speakers", str(min_speakers)])
        if max_speakers is not None:
            command.extend(["--max_speakers", str(max_speakers)])
        
        # Run transcribe.py with parameters
        subprocess.run(command, check=True)
        
        # 3. Find the latest JSON file in the output folder
        output_folder = "output"
        list_of_files = glob.glob(f"{output_folder}/*.json")
        if not list_of_files:
            raise HTTPException(status_code=404, detail="No JSON file found after transcription")
            
        latest_json = max(list_of_files, key=os.path.getctime)
        
        # 4. Run store_xchunk.py with the JSON file
        subprocess.run([python_executable, "store_chunk.py","--file", latest_json], check=True)
        
        # 5. Clean up
        os.remove(temp_audio_path)
        
        return {
            "status": "success",
            "message": "Audio processed and stored in vector database",
            "transcription_file": latest_json
        }
        
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)