from another import hello_world
import hub
import requests

# API endpoint
url = "https://api.restful-api.dev/objects"

# Send GET request
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Print the data
    print("Request successful!")
    print(f"Number of objects: {len(data)}")
    print("First few objects:")
    for obj in data[:3]:  # Show first 3 objects
        print(f"  - {obj}")
else:
    print(f"Request failed with status code: {response.status_code}")
    print(f"Response: {response.text}")

