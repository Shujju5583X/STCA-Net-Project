"""
STCA-Net Server Entry Point
Used by Dockerfile and Render deployment.
"""
from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, threaded=False, debug=False)
