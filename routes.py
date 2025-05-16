import os
import requests
from jose import jwt
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()  # This loads variables from a .env file into os.environ

CLERK_API_KEY = os.environ.get("CLERK_API_KEY")
CLERK_ISSUER = os.environ.get("CLERK_ISSUER")

app = Flask(__name__)

# Get these from your Clerk Dashboard
CLERK_API_KEY = os.environ.get("CLERK_API_KEY")
CLERK_ISSUER = os.environ.get("CLERK_ISSUER")  # e.g., https://your-app.clerk.accounts.dev
FLASK_SERVICE_ID = os.environ.get("FLASK_SERVICE_ID")

def create_service_jwt():
    """Create a JWT token for service-to-service authentication"""
    now = datetime.utcnow()
    payload = {
        "iss": CLERK_ISSUER,
        "sub": FLASK_SERVICE_ID,  # Unique identifier for your service
        "iat": now,
        "exp": now + timedelta(hours=1),
        # Add custom claims as needed
        "service": True,
        "name": "flask-backend-service"
    }
    
    # Get signing key from Clerk
    headers = {"Authorization": f"Bearer {CLERK_API_KEY}"}
    response = requests.get(f"{CLERK_ISSUER}/.well-known/jwks.json", headers=headers)
    jwks = response.json()
    
    # Create token
    token = jwt.encode(payload, jwks["keys"][0], algorithm="RS256")
    return token

@app.route("/make-authenticated-request")
def make_request():
    token = create_service_jwt()
    # Now use this token in your request to Next.js
    # next_js_api_url = "https://your-nextjs-app.com/api/protected-route"
    # headers = {"Authorization": f"Bearer {token}"}
    # response = requests.get(next_js_api_url, headers=headers)
    # return jsonify(response.json())
    print(f"JWT: {token}" )
    return "Done"

# Add this for development
if __name__ == "__main__":
    app.run(debug=True)


