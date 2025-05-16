
from flask import Flask, request, jsonify
import os
import json
import tempfile
import shutil
import requests
import google.generativeai as genai
import uuid
import zipfile
import mimetypes
from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey, DateTime, Text, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import enum
from werkzeug.utils import secure_filename
import logging
from sqlalchemy.sql import func
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)

# Configure Database
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/proofai")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Configure Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your-api-key")
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Piston API
PISTON_URL = os.environ.get("PISTON_URL", "http://localhost:2000")

# Configure NextJS API
NEXTJS_URL = os.environ.get("NEXTJS_URL", "http://host.docker.internal:3000")

# Database Models
class User(Base):
    __tablename__ = "User"
    
    id = Column(String, primary_key=True)
    clerkId = Column(String, unique=True)
    name = Column(String)
    email = Column(String, unique=True)
    imageUrl = Column(String, nullable=True)
    role = Column(String)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)


class Agent(Base):
    __tablename__ = 'agent'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False) 
    version = Column(String, nullable=False)
    isVerified = Column(Boolean, default=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    creatorId = Column(String, ForeignKey('user.id'))
    
    files = relationship("AgentFile", back_populates="agent", cascade="all, delete-orphan")
    tags = relationship("AgentTag", back_populates="agent", cascade="all, delete-orphan")
    envVars = relationship("AgentEnvVar", back_populates="agent", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="agent")


class AgentFile(Base):
    __tablename__ = 'agentfile'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    filesize = Column(Integer, nullable=False)
    mimetype = Column(String, nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    agentId = Column(String, ForeignKey('agent.id', ondelete='CASCADE'))
    
    agent = relationship("Agent", back_populates="files")


class AgentTag(Base):
    __tablename__ = 'agenttag'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    agentId = Column(String, ForeignKey('agent.id', ondelete='CASCADE'))
    
    agent = relationship("Agent", back_populates="tags")


class AgentEnvVar(Base):
    __tablename__ = "AgentEnvVar"
    
    id = Column(String, primary_key=True)
    key = Column(String)
    value = Column(String)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)
    agentId = Column(String, ForeignKey("Agent.id"))
    
    agent = relationship("Agent", back_populates="envVars")


class Session(Base):
    __tablename__ = "Session"
    
    id = Column(String, primary_key=True)
    startedAt = Column(DateTime)
    endedAt = Column(DateTime, nullable=True)
    userId = Column(String, ForeignKey("User.id"))
    agentId = Column(String, ForeignKey("Agent.id"))
    
    userVars = relationship("UserVariable", back_populates="session")
    chatHistory = relationship("ChatMessage", back_populates="session")
    jobs = relationship("Job", back_populates="session")


class UserVariable(Base):
    __tablename__ = "UserVariable"
    
    id = Column(String, primary_key=True)
    key = Column(String)
    value = Column(String)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)
    sessionId = Column(String, ForeignKey("Session.id"))
    
    session = relationship("Session", back_populates="userVars")


class ChatMessage(Base):
    __tablename__ = "ChatMessage"
    
    id = Column(String, primary_key=True)
    role = Column(String)  # 'user', 'agent', 'system'
    content = Column(String)
    timestamp = Column(DateTime)
    sessionId = Column(String, ForeignKey("Session.id"))
    
    session = relationship("Session", back_populates="chatHistory")



class JobLog(Base):
    __tablename__ = 'joblog'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    level = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    jobId = Column(String, ForeignKey('job.id', ondelete='CASCADE'))
    
    job = relationship("Job", back_populates="logs")



# Define enums for SQLAlchemy
class JobType(enum.Enum):
    AGENT_CREATE = "AGENT_CREATE"
    AGENT_UPDATE = "AGENT_UPDATE"
    MODEL_CREATE = "MODEL_CREATE"
    MODEL_UPDATE = "MODEL_UPDATE"
    DATASET_CREATE = "DATASET_CREATE"
    DATASET_UPDATE = "DATASET_UPDATE"
    AGENT_INVOCATION = "AGENT_INVOCATION"

class JobStatus(enum.Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Job(Base):
    __tablename__ = 'job'
    
    id = Column(String, primary_key=True)
    type = Column(Enum(JobType), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED)
    progress = Column(Float, default=0)
    statusMessage = Column(String, nullable=True)
    errorMessage = Column(String, nullable=True)
    createdAt = Column(DateTime, default=datetime.utcnow)
    startedAt = Column(DateTime, nullable=True)
    completedAt = Column(DateTime, nullable=True)
    userId = Column(String, ForeignKey('user.id'))
    agentId = Column(String, ForeignKey('agent.id'), nullable=True)
    datasetId = Column(String, ForeignKey('dataset.id'), nullable=True)
    modelId = Column(String, ForeignKey('model.id'), nullable=True)
    sessionId = Column(String, ForeignKey('session.id'), nullable=True)
    
    logs = relationship("JobLog", back_populates="job", cascade="all, delete-orphan")
    dataset = relationship("Dataset", back_populates="jobs")
    model = relationship("Model", back_populates="jobs")
    agent = relationship("Agent", back_populates="jobs")


class Dataset(Base):
    __tablename__ = 'dataset'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    version = Column(String, nullable=False)
    isVerified = Column(Boolean, default=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    creatorId = Column(String, ForeignKey('user.id'))
    
    files = relationship("DatasetFile", back_populates="dataset", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="dataset")

class DatasetFile(Base):
    __tablename__ = 'datasetfile'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    filesize = Column(Integer, nullable=False)
    mimetype = Column(String, nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    datasetId = Column(String, ForeignKey('dataset.id', ondelete='CASCADE'))
    
    dataset = relationship("Dataset", back_populates="files")

class Model(Base):
    __tablename__ = 'model'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    version = Column(String, nullable=False)
    isVerified = Column(Boolean, default=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    creatorId = Column(String, ForeignKey('user.id'))
    
    files = relationship("ModelFile", back_populates="model", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="model")

class ModelFile(Base):
    __tablename__ = 'modelfile'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    filesize = Column(Integer, nullable=False)
    mimetype = Column(String, nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    modelId = Column(String, ForeignKey('model.id', ondelete='CASCADE'))
    
    model = relationship("Model", back_populates="files")



# class JobLog(Base):
#     __tablename__ = "JobLog"
#
#     id = Column(String, primary_key=True)
#     level = Column(String)
#     message = Column(String)
#     timestamp = Column(DateTime)
#     jobId = Column(String, ForeignKey("Job.id"))
#
#     job = relationship("Job", back_populates="logs")
#

def create_proofai_context(session_id: str, db_session) -> Dict[str, Any]:
    """Create the context object for the proofai module"""
    session = db_session.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise ValueError(f"Session {session_id} not found")
    
    # Get environment variables
    env_vars = {}
    agent_env_vars = db_session.query(AgentEnvVar).filter(AgentEnvVar.agentId == session.agentId).all()
    for var in agent_env_vars:
        env_vars[var.key] = var.value
    
    # Get user variables
    user_vars = {}
    session_user_vars = db_session.query(UserVariable).filter(UserVariable.sessionId == session_id).all()
    for var in session_user_vars:
        user_vars[var.key] = var.value
    
    # Get chat history
    chat_history = []
    messages = db_session.query(ChatMessage).filter(
        ChatMessage.sessionId == session_id
    ).order_by(ChatMessage.timestamp).all()
    
    for msg in messages:
        chat_history.append({
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat()
        })
    
    return {
        "env_vars": env_vars,
        "user_vars": user_vars,
        "chat_history": chat_history,
        "session_id": session_id,
        "agent_id": session.agentId,
        "user_id": session.userId
    }


def generate_proofai_file(context: Dict[str, Any]) -> str:
    """Generate the proofai file with injected context"""
    context_template = """# context.py - auto-generated for ProofAI
# DO NOT EDIT THIS FILE

import os
import json
import requests
from typing import Dict, List, Any, Optional

_context = {
    "env_vars": %s,
    "user_vars": %s,
    "chat_history": %s,
    "session_id": "%s",
    "agent_id": "%s",
    "user_id": "%s"
}

def _init(context):
    global _context
    _context = context

def get_env_vars():
    return _context.get("env_vars", {})

def get_user_vars():
    return _context.get("user_vars", {})

def get_chat_history():
    return _context.get("chat_history", [])

def send_message(message):
    try:
        # First, print it for Piston to capture
        print({"type": "message", "content": message})
        
        # Then, save to database via the nextjs API
        api_url = "%s/api/sessions/" + _context["session_id"] + "/messages"
        payload = {
            "role": "agent",
            "content": message
        }
        response = requests.post(api_url, json=payload)
        return response.json()

    except Exception as e:
        print({"type": "error", "content": f"Error sending message: {str(e)}"})

def call_agent(agent_id, input_data):
    try:
        # First, print it for Piston to capture
        print({
            "type": "call_agent",
            "agent_id": agent_id,
            "input": input_data
        })
        
        # Then, call the agent via the nextjs API
        api_url = "%s/api/agents/" + agent_id + "/invoke"
        payload = {
            "prompt": input_data,
            "sessionId": _context["session_id"]
        }
        response = requests.post(api_url, json=payload)
        return response.json()
    except Exception as e:
        print({"type": "error", "content": f"Error calling agent: {str(e)}"})
        return {"error": str(e)}

def call_llm(prompt):
    try:
        import google.generativeai as genai
        
        # Configure the Gemini API key
        api_key = os.environ.get("GOOGLE_API_KEY", "%s")
        genai.configure(api_key=api_key)
        
        # Use Gemini Pro model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        print({"type": "error", "content": f"Error calling LLM: {str(e)}"})
        return f"Error calling LLM: {str(e)}"
""" % (
    json.dumps(context["env_vars"]),
    json.dumps(context["user_vars"]),
    json.dumps(context["chat_history"]),
    context["session_id"],
    context["agent_id"],
    context["user_id"],
    NEXTJS_URL,
    NEXTJS_URL,
    GOOGLE_API_KEY
)
    
    return context_template


def generate_init_file() -> str:
    """Generate the __init__.py file for the proofai module"""
    init_file = """# hub/__init__.py

from .context import (
    _init,
    get_env_vars,
    get_user_vars,
    get_chat_history,
    send_message,
    call_agent,
    call_llm
)
"""
    return init_file


def execute_with_piston(agent_files: List[Dict], temp_dir: str) -> Dict:
    """Execute the agent code using Piston"""
    # Prepare the payload for Piston
    payload = {
        "language": "python",
        "version": "3.11.0",
        "files": [
            {"name": f.get("filename", "main.py"), "content": f.get("content", "")}
            for f in agent_files
        ],
        "stdin": "",
        "args": ["run"],
        "compile_timeout": 10000,
        "run_timeout": 30000,
        "compile_memory_limit": -1,
        "run_memory_limit": -1
    }
    print("Payload")
    print(json.dumps(payload, indent=4))
    
    try:
        # Execute code with Piston
        response = requests.post(f"{PISTON_URL}/api/v2/execute", json=payload)
        print("sent the response")
        result = response.json()
        print("Running Piston")
        print(result)
        # Check if run is None and provide a default dictionary if it is
        if "run" not in result or result["run"] is None:
            result["run"] = {
                "stdout": "",
                "stderr": "Error: No run result returned from Piston",
                "code": 1,
                "signal": None,
                "output": "Error: No run result returned from Piston"
            }
        return result
    except Exception as e:
        return {
            "run": {
                "stdout": "",
                "stderr": f"Error executing with Piston: {str(e)}",
                "code": 1,
                "signal": None,
                "output": f"Error executing with Piston: {str(e)}"
            },
            "language": "python",
            "version": "3.10"
        }


def log_job_execution(job_id: str, message: str, level: str, db_session) -> None:
    """Add a log entry for the job"""
    log = JobLog(
        id=generate_uuid(),
        level=level,
        message=message,
        timestamp=datetime.now(),
        jobId=job_id
    )
    db_session.add(log)
    db_session.commit()


def update_job_status(job_id: str, status: str, message: str, progress: float, db_session) -> None:
    """Update the status of a job"""
    job = db_session.query(Job).filter(Job.id == job_id).first()
    if job:
        job.status = status
        job.statusMessage = message
        job.progress = progress
        
        if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            job.completedAt = datetime.now()
        elif status == "PROCESSING" and not job.startedAt:
            job.startedAt = datetime.now()
            
        db_session.commit()


def generate_uuid() -> str:
    """Generate a UUID for database records"""
    import uuid
    return str(uuid.uuid4())


def get_agent_files(agent_id: str, db_session) -> List[Dict]:
    """Get all files for an agent, ensuring main.py is first in the list if present"""
    agent_files = db_session.query(AgentFile).filter(AgentFile.agentId == agent_id).all()
    
    # Read file contents from the file system
    file_contents = []
    main_py_content = None
    
    for file in agent_files:
        try:
            with open(file.filepath, 'r') as f:
                content = f.read()
                file_data = {
                    "filename": file.filename,
                    "content": content
                }
                
                # If this is main.py, store it separately to insert at the beginning later
                if file.filename == "main.py":
                    main_py_content = file_data
                else:
                    file_contents.append(file_data)
        except Exception as e:
            app.logger.error(f"Error reading file {file.filepath}: {str(e)}")
    
    # If main.py was found, insert it at the beginning of the list
    if main_py_content:
        file_contents.insert(0, main_py_content)
    
    return file_contents


def add_chat_message(session_id: str, role: str, content: str, db_session) -> None:
    """Add a new chat message to the database"""
    message = ChatMessage(
        id=generate_uuid(),
        role=role,
        content=content,
        timestamp=datetime.now(),
        sessionId=session_id
    )
    db_session.add(message)
    db_session.commit()


@app.route('/api/models/create', methods=['POST'])
def create_model():
    """Handle model upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'File must have a name'}), 400
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'File must be a zip file'}), 400
    
    # Get parameters
    model_id = request.form.get('model_id', request.form.get('dataset_id'))  # Accept both for compatibility
    job_id = request.form.get('job_id')
    user_id = request.form.get('user_id')
    
    if not all([model_id, job_id, user_id]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Create database session
    db_session = SessionLocal()
    
    try:
        # Check if job exists
        job = db_session.query(Job).filter(Job.id == job_id).first()
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Update job status to processing
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Processing model upload", progress=10)
        
        # Save zip file temporarily
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp', f"{model_id}.zip")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Extract metadata
        metadata, error = extract_metadata(temp_path)
        if not metadata:
            return jsonify({'error': 'Folder must have metadata'}), 400
        if error:
            update_job_status2(db_session, job_id, JobStatus.FAILED, error=error)
            return jsonify({'error': error}), 400
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Metadata extracted successfully", progress=30)
        
        # Check if model exists
        model = db_session.query(Model).filter(Model.id == model_id).first()
        if not model:
            # Create new model
            model = Model(
                id=model_id,
                name=metadata['name'],
                description=metadata['description'],
                version=metadata['version'],
                creatorId=user_id
            )
            db_session.add(model)
        else:
            # Update existing model
            model.name = metadata['name']
            model.description = metadata['description']
            model.version = metadata['version']
            # model.updatedAt = datetime.utcnow()
        
        # Process zip file contents
        file_info, error = process_zip_file(temp_path, model_id, 'models')
        if error:
            update_job_status2(db_session, job_id, JobStatus.FAILED, error=error)
            return jsonify({'error': error}), 400
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Files extracted successfully", progress=60)
        
        # Create file entries
        if not file_info:
            return jsonify({'error': 'Missing file info'}), 400
        for file_data in file_info:
            model_file = ModelFile(
                id=str(uuid.uuid4()),
                filename=file_data['filename'],
                filepath=file_data['filepath'],
                filesize=file_data['filesize'],
                mimetype=file_data['mimetype'],
                modelId=model_id
            )
            db_session.add(model_file)
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "File metadata stored", progress=80)
        
        # Commit changes
        db_session.commit()
        
        # Update job status to completed
        update_job_status2(db_session, job_id, JobStatus.SUCCEEDED, "Model upload completed successfully", progress=100)
        
        # Remove temporary file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'message': 'Model created successfully',
            'model_id': model_id
        })
    
    except Exception as e:
        db_session.rollback()
        error_msg = f"Error processing model: {str(e)}"
        logger.error(error_msg)
        update_job_status2(db_session, job_id, JobStatus.FAILED, error=error_msg)
        return jsonify({'error': error_msg}), 500
    
    finally:
        db_session.close()


@app.route('/api/files/content', methods=['GET'])
def get_file_content():
    """
    Endpoint to retrieve file contents
    Required query parameters:
    - file_id: ID of the file to retrieve
    - file_type: Type of file ('agent', 'dataset', or 'model')
    """
    try:
        # Get parameters
        file_id = request.args.get('file_id')
        file_type = request.args.get('file_type')
        
        # Validate parameters
        if not file_id:
            return jsonify({'error': 'Missing required parameter: file_id'}), 400
            
        if not file_type or file_type not in ['agent', 'dataset', 'model']:
            return jsonify({'error': 'Invalid or missing file_type. Must be one of: agent, dataset, model'}), 400
        
        # Create database session
        db_session = SessionLocal()
        
        try:
            # Query for the appropriate file based on file_type
            file_record = None
            
            if file_type == 'agent':
                file_record = db_session.query(AgentFile).filter(AgentFile.id == file_id).first()
            elif file_type == 'dataset':
                file_record = db_session.query(DatasetFile).filter(DatasetFile.id == file_id).first()
            elif file_type == 'model':
                file_record = db_session.query(ModelFile).filter(ModelFile.id == file_id).first()
            
            if not file_record:
                return jsonify({'error': f'{file_type.capitalize()} file not found'}), 404
            
            # Check if file exists on disk
            if not os.path.exists(file_record.filepath):
                return jsonify({'error': 'File not found on server'}), 404
            
            # Read file content
            try:
                # Handle binary files differently from text files
                mimetype = file_record.mimetype
                is_text = mimetype.startswith('text/') or mimetype in [
                    'application/json', 
                    'application/javascript',
                    'application/xml',
                    'application/python',
                    'application/x-python',
                    'application/x-javascript'
                ]
                
                if is_text:
                    # Read as text
                    with open(file_record.filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    return jsonify({
                        'success': True,
                        'filename': file_record.filename,
                        'file_id': file_id,
                        'file_type': file_type,
                        'mimetype': file_record.mimetype,
                        'content': content,
                        'encoding': 'utf-8'
                    })
                else:
                    # For binary files, return base64 encoded content
                    import base64
                    with open(file_record.filepath, 'rb') as f:
                        binary_content = f.read()
                    
                    encoded_content = base64.b64encode(binary_content).decode('ascii')
                    
                    return jsonify({
                        'success': True,
                        'filename': file_record.filename,
                        'file_id': file_id,
                        'file_type': file_type,
                        'mimetype': file_record.mimetype,
                        'content': encoded_content,
                        'encoding': 'base64'
                    })
                    
            except Exception as e:
                error_msg = f"Error reading file: {str(e)}"
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 500
                
        except Exception as e:
            error_msg = f"Database error: {str(e)}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
            
        finally:
            db_session.close()
            
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


@app.route('/api/files/list', methods=['GET'])
def list_files():
    """
    Endpoint to list files associated with an entity
    Required query parameters:
    - entity_id: ID of the entity (agent, dataset, or model)
    - entity_type: Type of entity ('agent', 'dataset', or 'model')
    """
    try:
        # Get parameters
        entity_id = request.args.get('entity_id')
        entity_type = request.args.get('entity_type')
        
        # Validate parameters
        if not entity_id:
            return jsonify({'error': 'Missing required parameter: entity_id'}), 400
            
        if not entity_type or entity_type not in ['agent', 'dataset', 'model']:
            return jsonify({'error': 'Invalid or missing entity_type. Must be one of: agent, dataset, model'}), 400
        
        # Create database session
        db_session = SessionLocal()
        
        try:
            # Query for files based on entity_type
            files = []
            file_records = []
            
            if entity_type == 'agent':
                file_records = db_session.query(AgentFile).filter(AgentFile.agentId == entity_id).all()
            elif entity_type == 'dataset':
                file_records = db_session.query(DatasetFile).filter(DatasetFile.datasetId == entity_id).all()
            elif entity_type == 'model':
                file_records = db_session.query(ModelFile).filter(ModelFile.modelId == entity_id).all()
            
            for file_record in file_records:
                files.append({
                    'id': file_record.id,
                    'filename': file_record.filename,
                    'filesize': file_record.filesize,
                    'mimetype': file_record.mimetype,
                    'created_at': file_record.createdAt.isoformat() if hasattr(file_record, 'createdAt') else None
                })
            
            return jsonify({
                'success': True,
                'entity_id': entity_id,
                'entity_type': entity_type,
                'files': files,
                'count': len(files)
            })
                
        except Exception as e:
            error_msg = f"Database error: {str(e)}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
            
        finally:
            db_session.close()
            
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    # Ensure all directories exist
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'agents'), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'models'), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'temp'), exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0')

