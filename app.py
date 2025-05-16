from flask import Flask, request, jsonify
import os
import json
import requests
import base64
import traceback
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import google.generativeai as genai
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Database connection parameters
DB_URL = os.getenv("DATABASE_URL")

# Piston API endpoint (self-hosted)
PISTON_URL = os.getenv("PISTON_URL", "http://localhost:2000/api/v2/execute")

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Connect to the database
def get_db_connection():
    try:
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

# Get agent details from database
def get_agent(agent_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM "Agent" WHERE id = %s
                """,
                (agent_id,)
            )
            return cur.fetchone()

# Get agent environment variables
def get_agent_env_vars(agent_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT key, value FROM "AgentEnvVar" WHERE "agentId" = %s
                """,
                (agent_id,)
            )
            env_vars = {}
            for row in cur.fetchall():
                env_vars[row['key']] = row['value']
            return env_vars

# Get agent files
def get_agent_files(agent_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, filename, filepath, filesize, mimetype 
                FROM "AgentFile" WHERE "agentId" = %s
                """,
                (agent_id,)
            )
            return cur.fetchall()

# Get user variables for a session
def get_user_vars(session_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT key, value FROM "UserVariable" WHERE "sessionId" = %s
                """,
                (session_id,)
            )
            user_vars = {}
            for row in cur.fetchall():
                user_vars[row['key']] = row['value']
            return user_vars

# Set a user variable for a session
def set_user_var(session_id, key, value):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check if variable exists
            cur.execute(
                """
                SELECT id FROM "UserVariable" 
                WHERE "sessionId" = %s AND key = %s
                """,
                (session_id, key)
            )
            row = cur.fetchone()
            
            if row:
                # Update existing variable
                cur.execute(
                    """
                    UPDATE "UserVariable" 
                    SET value = %s, "updatedAt" = NOW()
                    WHERE "sessionId" = %s AND key = %s
                    """,
                    (value, session_id, key)
                )
            else:
                # Insert new variable
                cur.execute(
                    """
                    INSERT INTO "UserVariable" (key, value, "sessionId")
                    VALUES (%s, %s, %s)
                    """,
                    (key, value, session_id)
                )
            
            conn.commit()

# Get chat history for a session
def get_chat_history(session_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, role, content, timestamp FROM "ChatMessage" 
                WHERE "sessionId" = %s
                ORDER BY timestamp ASC
                """,
                (session_id,)
            )
            history = []
            for row in cur.fetchall():
                history.append({
                    "id": row['id'],
                    "role": row['role'],
                    "content": row['content'],
                    "timestamp": row['timestamp'].isoformat() if row['timestamp'] else None
                })
            return history

# Add a message to chat history
def add_chat_message(session_id, role, content):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO "ChatMessage" (role, content, "sessionId")
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (role, content, session_id)
            )
            message_id = cur.fetchone()[0]
            conn.commit()
            return message_id

# Update job status
def update_job_status(job_id, status, progress=None, status_message=None, error_message=None):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            update_parts = ["status = %s"]
            update_values = [status]
            
            if progress is not None:
                update_parts.append("progress = %s")
                update_values.append(progress)
            
            if status_message is not None:
                update_parts.append("\"statusMessage\" = %s")
                update_values.append(status_message)
            
            if error_message is not None:
                update_parts.append("\"errorMessage\" = %s")
                update_values.append(error_message)
            
            # Set timestamps based on status
            if status == "PROCESSING":
                update_parts.append("\"startedAt\" = NOW()")
            elif status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                update_parts.append("\"completedAt\" = NOW()")
            
            set_clause = ", ".join(update_parts)
            
            query = f"""
            UPDATE "Job"
            SET {set_clause}
            WHERE id = %s
            """
            
            cur.execute(query, update_values + [job_id])
            conn.commit()

# Add job log
def add_job_log(job_id, level, message):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO "JobLog" (level, message, "jobId")
                VALUES (%s, %s, %s)
                """,
                (level, message, job_id)
            )
            conn.commit()

# Call Gemini API for LLM completion
def call_gemini(prompt):
    try:
        if not GEMINI_API_KEY:
            return {"error": "Gemini API key not configured", "content": "LLM not available"}
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate content
        response = model.generate_content(prompt)
        
        return {
            "content": response.text,
            "model": "gemini-pro",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return {
            "error": str(e),
            "content": "Error processing LLM request",
            "status": "error"
        }

# Read file content from storage
def read_file_content(filepath):
    # In a production environment, this would read from a file storage service
    # For now, we'll simulate this with sample content
    
    # Extract file extension
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() == '.py':
        if 'main' in filepath:
            return """
import hub

def run(prompt):
    # Get environment variables
    env = hub.get_env_vars()
    
    # Get user variables
    user_vars = hub.get_user_vars()
    
    # Get chat history
    history = hub.get_chat_history()
    
    # Send a message back to the user
    hub.send_message(f"Received prompt: {prompt}")
    
    # Call LLM
    llm_response = hub.call_llm(f"Respond to the following user request: {prompt}")
    
    # Send LLM response
    hub.send_message(f"Here's what I found: {llm_response}")
    
    return {"status": "success", "processed": True}
"""
        else:
            return "# Helper functions\n\ndef helper():\n    return 'Helper function called'"
    else:
        return f"Sample content for {filepath}"

# Route for agent invocation
@app.route('/api/agents/<agent_id>/invoke', methods=['POST'])
def invoke_agent(agent_id):
    try:
        # Parse the request body
        request_data = request.json
        session_id = request_data.get('sessionId')
        job_id = request_data.get('jobId')
        user_id = request_data.get('userId')
        prompt = request_data.get('prompt')
        
        logger.info(f"Received invocation request for agent {agent_id}, session {session_id}")
        
        # Validate request
        if not all([session_id, job_id, user_id, prompt]):
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Update job status to PROCESSING
        update_job_status(job_id, "PROCESSING", 10, "Starting agent execution")
        add_job_log(job_id, "info", "Agent invocation started")
        
        # Get agent details
        agent = get_agent(agent_id)
        if not agent:
            update_job_status(job_id, "FAILED", 0, "Agent not found", "Agent not found in database")
            add_job_log(job_id, "error", f"Agent with ID {agent_id} not found")
            return jsonify({"error": "Agent not found"}), 404
        
        # Get agent environment variables
        env_vars = get_agent_env_vars(agent_id)
        
        # Get user variables for the session
        user_vars = get_user_vars(session_id)
        
        # Get chat history for the session
        chat_history = get_chat_history(session_id)
        
        # Get agent files
        agent_files = get_agent_files(agent_id)
        
        # Prepare context for the agent
        context = {
            "env_vars": env_vars,
            "user_vars": user_vars,
            "chat_history": chat_history,
            "prompt": prompt,
            "agent_id": agent_id,
            "session_id": session_id,
            "job_id": job_id,
            "user_id": user_id
        }
        
        update_job_status(job_id, "PROCESSING", 30, "Preparing agent execution environment")
        add_job_log(job_id, "info", "Agent context prepared")
        
        # Prepare hub module with function implementations
        hub_module = """
import json
import sys

_context = {}

def _init(context):
    global _context
    _context = context

def get_env_vars():
    """Return agent environment variables."""
    return _context.get("env_vars", {})

def get_user_vars():
    """Return user variables for the current session."""
    return _context.get("user_vars", {})

def set_user_var(key, value):
    """Set a user variable for the current session."""
    print(json.dumps({"type": "set_user_var", "key": key, "value": value}))
    return True

def send_message(msg):
    """Send a message from the agent to the user."""
    print(json.dumps({"type": "message", "content": msg}))
    return True

def get_chat_history():
    """Return the conversation history."""
    return _context.get("chat_history", [])

def call_agent(agent_id, input_data):
    """Call another agent with the specified input data."""
    print(json.dumps({"type": "agent_call", "agent_id": agent_id, "input": input_data}))
    return {"status": "called", "agent_id": agent_id}

def call_llm(prompt):
    """Call the LLM (Google Gemini) with the specified prompt."""
    print(json.dumps({"type": "llm_call", "prompt": prompt}))
    # In a real execution environment, we would wait for the response
    # For now, we'll just return a placeholder
    return "LLM response placeholder for: " + prompt
"""
        
        # Find main file for the agent
        main_file = None
        agent_code_files = []
        
        for file in agent_files:
            if file['filename'] == 'main.py':
                main_file = file
            if file['mimetype'] == 'text/x-python':
                file_content = read_file_content(file['filepath'])
                agent_code_files.append({
                    "name": file['filename'],
                    "content": file_content
                })
        
        if not main_file and not agent_code_files:
            update_job_status(job_id, "FAILED", 0, "Agent files not found", "No Python files found for agent")
            add_job_log(job_id, "error", "No Python files found for agent")
            return jsonify({"error": "Agent files not found"}), 404
        
        # If no main.py was found, use the first Python file
        if not main_file and agent_code_files:
            main_file = {"filename": agent_code_files[0]["name"], "filepath": ""}
            main_content = agent_code_files[0]["content"]
        else:
            main_content = read_file_content(main_file['filepath'])
        
        # Prepare the code to be executed with context
        execution_code = f"""
# Context initialization
__context__ = {json.dumps(context)}

# Hub module
{hub_module}

# Initialize hub with context
_init(__context__)

# Main agent code
{main_content}

# Check if run function exists and call it with the prompt
if "run" in globals():
    try:
        result = run("{prompt.replace('"', '\\"')}")
        print(json.dumps({{"type": "result", "data": result}}))
    except Exception as e:
        import traceback
        error = traceback.format_exc()
        print(json.dumps({{"type": "error", "error": str(e), "traceback": error}}))
else:
    print(json.dumps({{"type": "error", "error": "No run function found in agent code"}}))
"""
        
        # Prepare Piston API payload
        piston_payload = {
            "language": "python3",
            "files": [
                {
                    "name": "main.py",
                    "content": execution_code
                }
            ],
            "stdin": "",
            "args": [],
            "run": True,
            "compile_timeout": 10000,
            "run_timeout": 10000,
            "memory_limit": 512000  # 512MB
        }
        
        update_job_status(job_id, "PROCESSING", 50, "Submitting to execution environment")
        add_job_log(job_id, "info", "Submitting code to Piston executor")
        
        try:
            # Call Piston API
            piston_response = requests.post(PISTON_URL, json=piston_payload, timeout=30)
            
            if piston_response.status_code != 200:
                error_message = f"Piston API returned status code {piston_response.status_code}"
                logger.error(error_message)
                update_job_status(job_id, "FAILED", 0, "Execution environment error", error_message)
                add_job_log(job_id, "error", error_message)
                
                try:
                    error_details = piston_response.json()
                    add_job_log(job_id, "error", f"Piston error details: {json.dumps(error_details)}")
                except:
                    pass
                    
                return jsonify({
                    "error": "Execution environment error",
                    "details": error_message
                }), 500
            
            piston_data = piston_response.json()
            update_job_status(job_id, "PROCESSING", 70, "Processing execution results")
            
            # Process execution result
            run_result = piston_data.get('run', {})
            stdout = run_result.get('stdout', '')
            stderr = run_result.get('stderr', '')
            
            # Log any stderr output
            if stderr:
                add_job_log(job_id, "warning", f"Agent stderr output: {stderr}")
            
            # Process stdout for messages and results
            messages = []
            result_data = None
            error_data = None
            
            for line in stdout.strip().split('\n'):
                if not line:
                    continue
                
                try:
                    msg_data = json.loads(line)
                    msg_type = msg_data.get('type')
                    
                    if msg_type == 'message':
                        # Add agent message to chat history
                        content = msg_data.get('content', '')
                        add_chat_message(session_id, 'agent', content)
                        messages.append({
                            "role": "agent", 
                            "content": content
                        })
                        add_job_log(job_id, "info", f"Agent message: {content[:100]}...")
                    
                    elif msg_type == 'set_user_var':
                        # Set user variable
                        key = msg_data.get('key')
                        value = msg_data.get('value')
                        if key and value is not None:
                            set_user_var(session_id, key, str(value))
                            add_job_log(job_id, "info", f"User variable set: {key}={value}")
                    
                    elif msg_type == 'llm_call':
                        # Process LLM call
                        llm_prompt = msg_data.get('prompt', '')
                        add_job_log(job_id, "info", f"LLM call with prompt: {llm_prompt[:100]}...")
                        
                        # Call Gemini
                        llm_response = call_gemini(llm_prompt)
                        
                        if 'error' in llm_response:
                            add_job_log(job_id, "warning", f"LLM call error: {llm_response['error']}")
                    
                    elif msg_type == 'agent_call':
                        # Process agent call
                        called_agent_id = msg_data.get('agent_id', '')
                        input_data = msg_data.get('input', {})
                        add_job_log(job_id, "info", f"Agent call requested to agent: {called_agent_id}")
                        # In a full implementation, you would handle agent-to-agent calls here
                    
                    elif msg_type == 'result':
                        # Final result from the agent
                        result_data = msg_data.get('data', {})
                        add_job_log(job_id, "info", f"Agent execution result: {json.dumps(result_data)}")
                    
                    elif msg_type == 'error':
                        # Error in agent execution
                        error_data = {
                            "error": msg_data.get('error', 'Unknown error'),
                            "traceback": msg_data.get('traceback', '')
                        }
                        add_job_log(job_id, "error", f"Agent execution error: {error_data['error']}")
                        add_job_log(job_id, "error", f"Traceback: {error_data['traceback']}")
                
                except json.JSONDecodeError:
                    # Handle non-JSON output
                    add_job_log(job_id, "warning", f"Non-JSON output from agent: {line}")
            
            if error_data:
                update_job_status(job_id, "FAILED", 100, "Agent execution error", error_data['error'])
                
                return jsonify({
                    "success": False,
                    "error": "Agent execution error",
                    "details": error_data['error'],
                    "agentId": agent_id,
                    "sessionId": session_id,
                    "jobId": job_id,
                    "messages": messages
                }), 200  # Return 200 even for agent execution errors
            
            # Update job status to succeeded
            update_job_status(job_id, "SUCCEEDED", 100, "Agent execution completed")
            
            return jsonify({
                "success": True,
                "agentId": agent_id,
                "sessionId": session_id,
                "jobId": job_id,
                "messages": messages,
                "result": result_data or {}
            })
        
        except requests.exceptions.RequestException as e:
            # Handle Piston API call errors
            error_message = f"Error calling Piston API: {str(e)}"
            logger.error(error_message)
            update_job_status(job_id, "FAILED", 0, "Execution environment error", error_message)
            add_job_log(job_id, "error", error_message)
            
            return jsonify({
                "error": "Execution environment error",
                "details": error_message
            }), 500
    
    except Exception as e:
        # Handle general errors
        error_message = str(e)
        traceback_str = traceback.format_exc()
        logger.error(f"Error invoking agent: {error_message}")
        logger.error(traceback_str)
        
        # Update job status if job_id was provided
        if 'job_id' in locals():
            update_job_status(job_id, "FAILED", 0, "Internal server error", error_message)
            add_job_log(job_id, "error", f"Internal server error: {error_message}")
            add_job_log(job_id, "error", f"Traceback: {traceback_str}")
        
        return jsonify({
            "error": "Failed to invoke agent",
            "details": error_message
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check database connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "version": "1.0.0"
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "version": "1.0.0"
        }), 500

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(debug=os.getenv('FLASK_ENV') == 'development', host='0.0.0.0', port=port)
