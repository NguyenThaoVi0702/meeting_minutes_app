from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import uvicorn

from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy.sql import func
from pydub import AudioSegment
import io
import httpx
from pydantic import BaseModel
from typing import Optional, List
import json
import shutil
import requests 
import aiohttp
from io import BytesIO

from utils import create_meeting_minutes_doc_buffer, process_filename, md_to_docx
from fastapi.responses import JSONResponse
from openai import AzureOpenAI, OpenAIError

from system_prompt import CHAT_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT_V2, CHAT_MESSAGE, GET_CONCLUSION_SYSTEM_PROMPT
from datetime import datetime
import uuid
import logging
import os
import io
from pathlib import Path 

# Define log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
temp_dir = "./audio_data"
os.makedirs(temp_dir, exist_ok=True)

MAX_ATTEMPT = 3
MAX_CONCURRENT_REQUESTS = 50
LIMIT_TURN = 6
client = None
engine = None

# Define schema for /chat endpoint
class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str

class Summary(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, max_length=255)
    session_id: Optional[str] = Field(default=None, max_length=255)
    filename: Optional[str] = Field(default=None, max_length=255)
    summary_text: str
    raw_text: str
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

class Chat(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, max_length=255)
    session_id: Optional[str] = Field(default=None, max_length=255)
    role: int = Field(foreign_key="role.id")
    message: str
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

class Role(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    role: str

class Segment(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None, max_length=25)
    session_id: Optional[str] = Field(default=None, max_length=255)
    ckpt: int
    file_name: Optional[str] = Field(default=None, max_length=255)
    transcript: str
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    # updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        sa_column_kwargs={"onupdate": func.now()},
        nullable=False
    )

class Conclusion(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    session_id: Optional[str] = Field(default=None, max_length=255)
    conclusion: str
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    # updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        sa_column_kwargs={"onupdate": func.now()},
        nullable=False
    )

app = FastAPI(title = 'AI Chat Server')
DATABASE_URL = f'postgresql+psycopg2://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}'

@app.on_event("startup")
def on_startup():
    global client
    global engine

    engine = create_engine(DATABASE_URL, echo=True)
    SQLModel.metadata.create_all(engine)
    print(f'Created Connection: {DATABASE_URL}')
    # Init AzureOpenAI client
    client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment='gpt-4.1-GS'
        )

# AI Agent chat
def ai_agent_process(user_message: str, task: str = 'chat', messages: List = []) -> str:
    if task == 'chat':
        system_prompt = CHAT_SYSTEM_PROMPT
    elif task == 'summary':
        system_prompt = SUMMARY_SYSTEM_PROMPT_V2
    elif task == 'get_conclusion':
        system_prompt = GET_CONCLUSION_SYSTEM_PROMPT
    try:
        send_message = [{"role": "system", "content": system_prompt}]
        if task != 'get_conclusion':
            for msg in messages[::-1]:
                send_message.append(msg)
        send_message.append({"role": "user" , "content": user_message})
        logger.info(f'Send messages: {send_message}')
        response = client.chat.completions.create(
                    model = 'gpt-4.1-GS',
                    # messages = [
                    #         {"role": "system", "content": system_prompt},
                    #         {"role": "user" , "content": user_message}
                    #     ],
                    messages = send_message,
                    temperature=0.2
        )
        response_text = response.choices[0].message.content
        return response_text
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail='Lỗi khi kết nối tới AI Agent.')
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail='Lỗi không xác dịnh khi xử lý yêu cầu.')

@app.post("/chat")
async def chat_endpoint(data: ChatRequest):
  try:
    summary_text = ''
    raw_text = ''
    try:
        """Get summary by session id"""
        with Session(engine) as session:
            # statement = select(Summary).where(Summary.session_id == data.session_id)
            statement = (
                select(Summary)
                .where(Summary.session_id == data.session_id)
                .where(Summary.user_id == data.user_id)
                .order_by(Summary.created_at.desc())
                .limit(1)
            )
            summary_record = session.exec(statement).first()
            summary_text = summary_record.summary_text
            raw_text = summary_record.raw_text
    
    except Exception as error:
        logger.error(f'Error querying database: {error}')

    messages = []
    try:
        """Get Chat by session id"""
        with Session(engine) as session:
            statement = (
                select(Chat,  Role.role)
                .join(Role, Chat.role == Role.id)
                .where(Chat.session_id == data.session_id)
                .order_by(Chat.created_at.desc())
                .limit(LIMIT_TURN)
            )
            results = session.exec(statement).all()

            for chat, role_value in results:
                # print(f"Message: {chat.message}, Role: {role_value}")
                messages.append({'role': role_value, 'content': chat.message})
    
    except Exception as error:
        logger.error(f'Error querying database: {error}')

    try:
        msg = CHAT_MESSAGE.format(raw_text = raw_text, summary_text = summary_text, user_msg=data.message)
    except Exception as error:
        logger.error(f'Got error - msg: {error}')
        msg = data.message
    ai_response = ai_agent_process(msg, task = 'chat', messages=messages)

    if ai_response.startswith('Đoạn tóm tắt:'):
        logger.info('Update Summary')
        ai_response = ai_response[len('Đoạn tóm tắt:'):].strip()

        try:
            """Save to Database"""
            summary_record.summary_text = ai_response
            with Session(engine) as session:
                session.add(summary_record)
                session.commit()
        except Exception as error:
            logger.error(f"Got error when save to DB: {error}")

    elif ai_response.startswith('Câu trả lời:'):
        logger.info('No update - Chat turn')
        ai_response = ai_response[len('Câu trả lời:'):].strip()

    try:
        session_id = data.session_id
        
        with Session(engine) as session:
            user_role = session.exec(select(Role).where(Role.role == "user")).first()
            assistant_role = session.exec(select(Role).where(Role.role == "assistant")).first()

            input_chat_record = Chat(user_id=data.user_id, session_id=session_id, message=data.message, role=user_role.id,)
            chat_record = Chat(user_id=data.user_id, session_id=session_id, message=ai_response, role=assistant_role.id)

            session.add(input_chat_record)
            session.add(chat_record)
            session.commit()
    except Exception as error:
       logger.error(f'Error saving database:  {error}')
    return {
          "status_code": 200,
          "user_id": data.user_id,
          "response": ai_response
          }
  except HTTPException as e:
    return {"status_code": e.status_code, "message": e.detail}
  except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return {"status_code": 500, "message": "Lỗi không xác d?nh."}

@app.post("/summary")
async def summary_endpoint(
        user_id: str = Form(...),
        session_id: str=Form(...),
        file: UploadFile = File(None) 
    ):
    try:
        text_content = ""
        if file:
            file_content = await file.read()
            text_content = file_content.decode("utf-8")
        else:
            logger.info("Không có file dính kèm. Sử dụng nội dung rỗng.")
        filename = session_id.split('/')[-1]
        ai_response = ai_agent_process(text_content, task = 'summary')
        try:
            """Save to Database"""
            summary_record = Summary(
                user_id = user_id, 
                session_id=session_id, 
                filename=filename, 
                summary_text=ai_response, 
                raw_text=text_content 
                )
            with Session(engine) as session:
                session.add(summary_record)
                session.commit()
        except Exception as error:
            logger.error(f"[Summary] Error saving database: {error}")
        return {
            "status_code": 200,
            "user_id": user_id,
            "summary": ai_response
        }
    except HTTPException as e:
        return {"status_code": e.status_code, "message": e.detail}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"status_code": 500, "message": f"Lỗi không xác định. {e}"}

@app.post("/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_id: Optional[str] = Form(None)
):
    filename = file.filename
    contents = await file.read()
    file_path = os.path.join(temp_dir, filename)
    with io.open(file_path, "wb") as buffer:
        buffer.write(contents)

    # Save to Database
    file_name, ckpt = process_filename(filename)

    segment = Segment(
        session_id= session_id,
        user_id = user_id,
        ckpt=int(ckpt),
        file_name= file_name,
        transcript= '',
    )

    with Session(engine) as session:
        session.add(segment)
        session.commit()
        session.refresh(segment)

    url = 'http://10.43.128.107/v1/audio/transcriptions'
    data = {
        'language': 'vi',
        'model': './models/whisper_large_v3_turbo_finetune'
    }
    text = ''
    with open(file_path, 'rb') as audio_file:
        form = aiohttp.FormData()
        form.add_field('file', audio_file, filename=filename, content_type='audio/wav')
        for k, v in data.items():
            form.add_field(k, v)

        async with aiohttp.ClientSession() as session:
            for times in range(MAX_ATTEMPT):
                async with session.post(url, data=form) as response:
                    if response.status == 200:
                        status_code = 200
                        json_data = await response.json()
                        text = json_data.get('text', '')
                        break
                    elif response.status == 500:
                        status_code = 500
                        text = ''
                    else:
                        print(f"Error: {response.status}, {response.text}")
                        status_code = 500
                        text = ''
                        break
        audio_file.close()
    # Update to Database /transcription
    with Session(engine) as session:
        segment = session.get(Segment, segment.id)
        if segment:
            segment.transcript = text
            session.commit()
        else:
            print(f"Error: Segment with id {segment.id} not found for update.")

    return {
            "status_code": status_code,
            "text": text
            }

@app.get("/get-summary")
async def get_summary(
    session_id: str=Form(...),
    user_id: str=Form(...)
    ):
    try:
        """Get summary by session id"""
        # filename = session_id.split('/')[-1]
        filename = Path(session_id).name 
        with Session(engine) as session:
            statement = (
                select(Summary)
                .where(Summary.user_id == user_id)
                .where(Summary.filename == filename)
                .order_by(Summary.created_at.desc())
                .limit(1)
            )
            summary_record = session.exec(statement).first()
            summary_text = summary_record.summary_text
            # raw_text = summary_record.raw_text
            return {
                "status_code": 200,
                "summary": summary_text
            }
    except Exception as error:
        error_msg = f'Error querying database: {error}'
        logger.error(error_msg)
        return {"status_code": 500, "message": error_msg}
        
@app.post("/download-word")
async def download_word(
    session_id: str = Form(...),
    user_id: str = Form(...)
    ):
    logger.info(f'Got Request: /download-word')
    filename = Path(session_id).name 
    try:
        """Get summary by session id"""
        with Session(engine) as session:
            # statement = select(Summary).where(Summary.session_id == session_id)
            statement = (
                select(Summary)
                .where(Summary.user_id == user_id)
                .where(Summary.filename == filename)
                .order_by(Summary.created_at.desc())
                .limit(1)
            )
            summary_record = session.exec(statement).first()
            summary_text = summary_record.summary_text
            # raw_text = summary_record.raw_text
    except Exception as error:
        error_msg = f'Error querying database: {error}'
        logger.error(error_msg)
        return {"status_code": 500, "message": error_msg}
    logger.info(f'Query Database: Done')
    buffer = create_meeting_minutes_doc_buffer(summary_text)

    try:
        buffer = create_meeting_minutes_doc_buffer(summary_text)
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=BienBanHop.docx"}
        )
    # except HTTPException as e:
    #     error_msg = ''
    #     raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")
    except Exception as e:
        # try:
        #     buffer = md_to_docx(markdown_file_path)
        #     return StreamingResponse(
        #         buffer,
        #         media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        #         headers={"Content-Disposition": "attachment; filename=BienBanHop.docx"}
        #     )
        # except Exception as errpr:
        #     pass
        error_msg = f"Unexpected error during DOCX generation or streaming: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=f"Internal server error: {error_msg}")

@app.post("/get-conclusion")
async def get_conclusion(session_id: str = Form(...)):
    logger.info(f'Got Request: /get-conclusion')
    try:
        """Get conclusions from summarry"""
        with Session(engine) as session:
            # statement = select(Summary).where(Summary.session_id == session_id)
            statement = (
                select(Summary)
                .where(Summary.session_id == session_id)
                .order_by(Summary.created_at.desc())
                .limit(1)
            )
            summary_record = session.exec(statement).first()
            summary_text = summary_record.summary_text
            # raw_text = summary_record.raw_text
    except Exception as error:
        error_msg = f'Error querying database: {error}'
        logger.error(error_msg)
        return {"status_code": 500, "message": error_msg}
    logger.info(f'Query Database: Done')
    ai_response = ai_agent_process(summary_text, task = 'get_conclusion')
    try:
        conclusion_record= Conclusion(
            session_id=session_id,
            conclusion=ai_response
        )
        with Session(engine) as session:
            session.add(conclusion_record)
            session.commit()
    except Exception as error:
        print(f'Error saving conclusion: {error}')
    return ai_response

@app.post("/assign-tasks")
async def assign_tasks(session_id: str = Form(...)):
    logger.info(f'Got Request: /assign-tasks')

    """Assign tasks from conclusions by session id"""
    
    # Query Databasa
    conclusion_text = ''
    try:
        with Session(engine) as session:
            statement = (
                select(Conclusion)
                .where(Conclusion.session_id == session_id)
                .order_by(Conclusion.created_at.desc())
                .limit(1)
            )
            conclusion_record = session.exec(statement).first()
            conclusion_text = conclusion_record.conclusion
    except Exception as error:
        print(f'Error quering database conclusion: {error}')
    msg = ''
    stt_code = 200
    if conclusion_text:
        try:
            url = "https://n8n-ai.vietinbank.vn/webhook/send-assign-tasks"
            payload = {"input_data": conclusion_text}
            response = requests.post(url, data=payload)
            msg = "Assign task: Done"
            stt_code = 200
        except Exception as e:
            logger.error(e)
            msg = "Assign task: Fail"
            stt_code = 500
    else:
        msg = "Assign task: Fail - Not found Conclusion"
        stt_code = 404
    return {"status_code": stt_code, "message": msg}

@app.post("/get-all-segment")
async def get_all_segment(
    session_id: str = Form(...),
    user_id: Optional[str] = Form(None)
    ):
    logger.info(f'Got Request: /get-all-segment')
    # pass
    try:
        file_name, ckpt = process_filename(session_id)
        with Session(engine) as session:
            if user_id is None:
                statement = (
                    select(Segment)
                    .where(Segment.file_name == file_name)
                    .order_by(Segment.ckpt.asc())
                )
            else:
                statement = (
                    select(Segment)
                    .where(Segment.file_name == file_name)
                    .where(Segment.user_id == user_id)
                    .order_by(Segment.ckpt.asc())
                )
        segment_records = session.exec(statement).all()
        full_text = ' '.join(segment.transcript for segment in segment_records)
        return {
                "status_code": 200,
                "full_text": full_text
                }
    except Exception as error:
        logger.error(f'Error querying database: {error}')


VLLM_API_URL = 'http://10.43.128.107:8085/v1/audio/transcriptions'  # Thay URL thực tế

@app.post("/convert-and-forward")
async def convert_and_forward(
    file: UploadFile = File(...),
    model: str = Form(...)
):

    if not file.filename.endswith(".webm"):
        raise HTTPException(status_code=400, detail="Only .webm files are supported")

    try:
        # Đọc file webm vào RAM
        webm_bytes = await file.read()
        webm_audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")

        # Export sang WAV (in-memory)
        wav_io = io.BytesIO()
        webm_audio.export(wav_io, format="wav")
        wav_io.seek(0)

        wav_bytes = wav_io.read()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting file: {str(e)}")

    # Gửi đến VLLM API
    try:

        files = {
            "file": wav_bytes
        }
        data = {'language': 'vi', 'model': (None, model)}

        async with httpx.AsyncClient() as client:
            response = await client.post(VLLM_API_URL, files=files, data = data)
        return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to forward to VLLM API: {str(e)}")

# Health check
@app.get("/status")
async def status():
  return {"status": "Server ok !"}

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port = 8000)
