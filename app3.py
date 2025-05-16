from flask import Flask, request, jsonify
import os
import uuid
import zipfile
import json
import mimetypes
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, ForeignKey, DateTime, Enum, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import enum
from werkzeug.utils import secure_filename
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure database
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost/agent_hub")
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

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

# Define SQLAlchemy models
class User(Base):
    __tablename__ = 'user'
    
    id = Column(String, primary_key=True)
    # Other user columns omitted for brevity

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
    __tablename__ = 'agentenvvar'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    agentId = Column(String, ForeignKey('agent.id', ondelete='CASCADE'))
    
    agent = relationship("Agent", back_populates="envVars")

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

class JobLog(Base):
    __tablename__ = 'joblog'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    level = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    jobId = Column(String, ForeignKey('job.id', ondelete='CASCADE'))
    
    job = relationship("Job", back_populates="logs")

# Define file storage path
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper functions
def get_file_mimetype(filepath):
    """Determine the MIME type of a file."""
    mime_type, _ = mimetypes.guess_type(filepath)
    return mime_type or "application/octet-stream"

def update_job_status2(db_session, job_id, status, message=None, error=None, progress=None):
    """Update job status and add log entry."""
    job = db_session.query(Job).filter(Job.id == job_id).first()
    if not job:
        logger.error(f"Job {job_id} not found")
        return False
    
    job.status = status
    
    if message:
        job.statusMessage = message
        # Add log entry
        log = JobLog(
            id=str(uuid.uuid4()),
            level="info",
            message=message,
            jobId=job_id
        )
        db_session.add(log)
    
    if error:
        job.errorMessage = error
        # Add error log entry
        log = JobLog(
            id=str(uuid.uuid4()),
            level="error",
            message=error,
            jobId=job_id
        )
        db_session.add(log)
    
    if progress is not None:
        job.progress = progress
    
    if status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
        job.completedAt = datetime.utcnow()
    
    db_session.commit()
    return True

def extract_metadata(zip_path):
    """Extract metadata.json from zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if 'metadata.json' not in zip_ref.namelist():
                return None, "metadata.json not found in zip file"
            
            with zip_ref.open('metadata.json') as metadata_file:
                metadata = json.load(metadata_file)
                required_fields = ['name', 'version', 'description']
                if not all(field in metadata for field in required_fields):
                    return None, f"metadata.json missing required fields: {', '.join(required_fields)}"
                
                return metadata, None
    except zipfile.BadZipFile:
        return None, "Invalid zip file"
    except json.JSONDecodeError:
        return None, "Invalid JSON in metadata.json"
    except Exception as e:
        return None, f"Error extracting metadata: {str(e)}"

def process_zip_file(zip_path, entity_id, entity_type):
    """Process zip file and extract contents."""
    entity_folder = os.path.join(UPLOAD_FOLDER, entity_type, entity_id)
    os.makedirs(entity_folder, exist_ok=True)
    
    file_info = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('/'):  # Skip directories
                    continue
                    
                # Extract file
                extracted_path = os.path.join(entity_folder, os.path.basename(file_name))
                with zip_ref.open(file_name) as source, open(extracted_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                
                # Get file info
                file_size = os.path.getsize(extracted_path)
                mime_type = get_file_mimetype(extracted_path)
                
                file_info.append({
                    'filename': os.path.basename(file_name),
                    'filepath': extracted_path,
                    'filesize': file_size,
                    'mimetype': mime_type
                })
        
        return file_info, None
    except Exception as e:
        return None, f"Error processing zip file: {str(e)}"

# API Routes
@app.route('/api/agents/create', methods=['POST'])
def create_agent():
    """Handle agent upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'File must be a zip file'}), 400
    
    # Get parameters
    agent_id = request.form.get('agent_id')
    job_id = request.form.get('job_id')
    user_id = request.form.get('user_id')
    
    if not all([agent_id, job_id, user_id]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Create database session
    db_session = SessionLocal()
    
    try:
        # Check if job exists
        job = db_session.query(Job).filter(Job.id == job_id).first()
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Update job status to processing
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Processing agent upload", progress=10)
        
        # Save zip file temporarily
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp', f"{agent_id}.zip")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Extract metadata
        metadata, error = extract_metadata(temp_path)
        if error:
            update_job_status2(db_session, job_id, JobStatus.FAILED, error=error)
            return jsonify({'error': error}), 400
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Metadata extracted successfully", progress=30)
        
        # Check if agent exists
        agent = db_session.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            # Create new agent
            agent = Agent(
                id=agent_id,
                name=metadata['name'],
                description=metadata['description'],
                version=metadata['version'],
                creatorId=user_id
            )
            db_session.add(agent)
        else:
            # Update existing agent
            agent.name = metadata['name']
            agent.description = metadata['description']
            agent.version = metadata['version']
            agent.updatedAt = datetime.utcnow()
        
        # Process zip file contents
        file_info, error = process_zip_file(temp_path, agent_id, 'agents')
        if error:
            update_job_status2(db_session, job_id, JobStatus.FAILED, error=error)
            return jsonify({'error': error}), 400
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Files extracted successfully", progress=60)
        
        # Create file entries
        for file_data in file_info:
            agent_file = AgentFile(
                id=str(uuid.uuid4()),
                filename=file_data['filename'],
                filepath=file_data['filepath'],
                filesize=file_data['filesize'],
                mimetype=file_data['mimetype'],
                agentId=agent_id
            )
            db_session.add(agent_file)
        
        # Process tags if included in metadata
        if 'tags' in metadata and isinstance(metadata['tags'], list):
            # Remove existing tags
            db_session.query(AgentTag).filter(AgentTag.agentId == agent_id).delete()
            
            # Add new tags
            for tag_name in metadata['tags']:
                tag = AgentTag(
                    id=str(uuid.uuid4()),
                    name=tag_name,
                    agentId=agent_id
                )
                db_session.add(tag)
        
        # Process environment variables if included in metadata
        if 'envVars' in metadata and isinstance(metadata['envVars'], dict):
            # Update existing env vars or create new ones
            for key, value in metadata['envVars'].items():
                env_var = db_session.query(AgentEnvVar).filter(
                    AgentEnvVar.agentId == agent_id,
                    AgentEnvVar.key == key
                ).first()
                
                if env_var:
                    env_var.value = value
                    env_var.updatedAt = datetime.utcnow()
                else:
                    env_var = AgentEnvVar(
                        id=str(uuid.uuid4()),
                        key=key,
                        value=value,
                        agentId=agent_id
                    )
                    db_session.add(env_var)
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "File metadata stored", progress=80)
        
        # Commit changes
        db_session.commit()
        
        # Update job status to completed
        update_job_status2(db_session, job_id, JobStatus.SUCCEEDED, "Agent upload completed successfully", progress=100)
        
        # Remove temporary file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'message': 'Agent created successfully',
            'agent_id': agent_id
        })
    
    except Exception as e:
        db_session.rollback()
        error_msg = f"Error processing agent: {str(e)}"
        logger.error(error_msg)
        update_job_status2(db_session, job_id, JobStatus.FAILED, error=error_msg)
        return jsonify({'error': error_msg}), 500
    
    finally:
        db_session.close()

@app.route('/api/datasets/create', methods=['POST'])
def create_dataset():
    """Handle dataset upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'File must be a zip file'}), 400
    
    # Get parameters
    dataset_id = request.form.get('dataset_id')
    job_id = request.form.get('job_id')
    user_id = request.form.get('user_id')
    
    if not all([dataset_id, job_id, user_id]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Create database session
    db_session = SessionLocal()
    
    try:
        # Check if job exists
        job = db_session.query(Job).filter(Job.id == job_id).first()
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Update job status to processing
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Processing dataset upload", progress=10)
        
        # Save zip file temporarily
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp', f"{dataset_id}.zip")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Extract metadata
        metadata, error = extract_metadata(temp_path)
        if error:
            update_job_status2(db_session, job_id, JobStatus.FAILED, error=error)
            return jsonify({'error': error}), 400
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Metadata extracted successfully", progress=30)
        
        # Check if dataset exists
        dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            # Create new dataset
            dataset = Dataset(
                id=dataset_id,
                name=metadata['name'],
                description=metadata['description'],
                version=metadata['version'],
                creatorId=user_id
            )
            db_session.add(dataset)
        else:
            # Update existing dataset
            dataset.name = metadata['name']
            dataset.description = metadata['description']
            dataset.version = metadata['version']
            dataset.updatedAt = datetime.utcnow()
        
        # Process zip file contents
        file_info, error = process_zip_file(temp_path, dataset_id, 'datasets')
        if error:
            update_job_status2(db_session, job_id, JobStatus.FAILED, error=error)
            return jsonify({'error': error}), 400
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Files extracted successfully", progress=60)
        
        # Create file entries
        for file_data in file_info:
            dataset_file = DatasetFile(
                id=str(uuid.uuid4()),
                filename=file_data['filename'],
                filepath=file_data['filepath'],
                filesize=file_data['filesize'],
                mimetype=file_data['mimetype'],
                datasetId=dataset_id
            )
            db_session.add(dataset_file)
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "File metadata stored", progress=80)
        
        # Commit changes
        db_session.commit()
        
        # Update job status to completed
        update_job_status2(db_session, job_id, JobStatus.SUCCEEDED, "Dataset upload completed successfully", progress=100)
        
        # Remove temporary file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'message': 'Dataset created successfully',
            'dataset_id': dataset_id
        })
    
    except Exception as e:
        db_session.rollback()
        error_msg = f"Error processing dataset: {str(e)}"
        logger.error(error_msg)
        update_job_status2(db_session, job_id, JobStatus.FAILED, error=error_msg)
        return jsonify({'error': error_msg}), 500
    
    finally:
        db_session.close()

@app.route('/api/models/create', methods=['POST'])
def create_model():
    """Handle model upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
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
            model.updatedAt = datetime.utcnow()
        
        # Process zip file contents
        file_info, error = process_zip_file(temp_path, model_id, 'models')
        if error:
            update_job_status2(db_session, job_id, JobStatus.FAILED, error=error)
            return jsonify({'error': error}), 400
        
        # Update job progress
        update_job_status2(db_session, job_id, JobStatus.PROCESSING, "Files extracted successfully", progress=60)
        
        # Create file entries
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

if __name__ == '__main__':
    # Ensure all directories exist
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'agents'), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'models'), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'temp'), exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0')
