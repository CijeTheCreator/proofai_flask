import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configuration
CLERK_API_KEY = os.environ.get("CLERK_API_KEY")
CLERK_API_URL = "https://api.clerk.dev/v1"

def get_clerk_session_token():
    """Create a session token using Clerk's API"""
    headers = {
        "Authorization": f"Bearer {CLERK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a session for a service user
    # Note: You should create a dedicated user for your service
    # through the Clerk Dashboard first
    service_user_id = os.environ.get("CLERK_SERVICE_USER_ID")
    
    if not service_user_id:
        raise ValueError("CLERK_SERVICE_USER_ID environment variable is required")

    
    # Create a session for this user
    url = f"{CLERK_API_URL}/users/{service_user_id}/sessions"
    response = requests.post(url, headers=headers)


    print(response.status_code)
    print(response.text)
    
    if response.status_code != 200:
        raise Exception(f"Failed to create session: {response.json()}")

    
    # Get the session token
    session_data = response.json()
    return session_data["client_token"]

@app.route("/")
def home():
    return jsonify({"message": "Flask backend service is running"})

@app.route("/make-authenticated-request")
def make_request():
    try:
        # Get a session token from Clerk
        token = get_clerk_session_token()
        print(f"token: {token}")
        
        # # Make request to Next.js
        # headers = {"Authorization": f"Bearer {token}"}
        # response = requests.get(NEXTJS_API_URL, headers=headers)
        #
        # return jsonify({
        #     "status": "success",
        #     "next_js_response": response.json(),
        #     "status_code": response.status_code
        # })
        #
        
        return jsonify({
            "status": "success",
            "next_js_response": "something",
            "status_code": "something_else"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Add this for development
if __name__ == "__main__":
    app.run(debug=True)
