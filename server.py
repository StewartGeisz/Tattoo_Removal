#!/usr/bin/env python3
"""
Simple HTTP Server for Manual Video Blur Tool
Serves the HTML file and video files locally
"""

import http.server
import socketserver
import os
import mimetypes
from urllib.parse import unquote

class VideoHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to properly serve video files with CORS headers"""
    
    def end_headers(self):
        # Add CORS headers for cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def guess_type(self, path):
        """Guess content type with proper video MIME types"""
        mimetype, _ = mimetypes.guess_type(path)
        
        # Ensure proper MIME types for video files
        if path.lower().endswith('.mp4'):
            return 'video/mp4'
        elif path.lower().endswith('.avi'):
            return 'video/x-msvideo'
        elif path.lower().endswith('.mov'):
            return 'video/quicktime'
        elif path.lower().endswith('.wmv'):
            return 'video/x-ms-wmv'
        elif path.lower().endswith('.webm'):
            return 'video/webm'
        
        return mimetype or 'application/octet-stream'
    
    def do_GET(self):
        """Handle GET requests with proper video streaming support"""
        # Decode URL
        path = unquote(self.path)
        
        # Serve the main HTML file at root
        if path == '/' or path == '':
            self.path = '/manual_blur_tool.html'
        
        # Handle range requests for video streaming
        if path.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.webm')):
            self.handle_video_request(path)
        else:
            super().do_GET()
    
    def handle_video_request(self, path):
        """Handle video requests with range support for streaming"""
        # Remove leading slash and get full file path
        file_path = path.lstrip('/')
        
        if not os.path.exists(file_path):
            self.send_error(404, "File not found")
            return
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Handle range requests (for video seeking)
        range_header = self.headers.get('Range')
        if range_header:
            # Parse range header
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1
            
            # Ensure end doesn't exceed file size
            end = min(end, file_size - 1)
            content_length = end - start + 1
            
            # Send partial content response
            self.send_response(206, 'Partial Content')
            self.send_header('Content-Type', self.guess_type(file_path))
            self.send_header('Content-Length', str(content_length))
            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            
            # Send the requested byte range
            try:
                with open(file_path, 'rb') as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(8192, remaining)
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (ConnectionResetError, BrokenPipeError):
                # Client disconnected, ignore
                pass
        else:
            # Send full file
            self.send_response(200)
            self.send_header('Content-Type', self.guess_type(file_path))
            self.send_header('Content-Length', str(file_size))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            
            try:
                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (ConnectionResetError, BrokenPipeError):
                # Client disconnected, ignore
                pass

def start_server(port=8000):
    """Start the local HTTP server"""
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("🎬 MANUAL VIDEO BLUR TOOL SERVER")
    print("=" * 60)
    print(f"📂 Serving from: {os.getcwd()}")
    print(f"🌐 Server running on: http://localhost:{port}")
    print("=" * 60)
    print("📋 Available videos:")
    
    # List available input videos
    input_dir = "input"
    if os.path.exists(input_dir):
        videos = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.webm'))]
        for i, video in enumerate(videos, 1):
            print(f"   {i}. {video}")
    else:
        print("   ❌ No input directory found")
    
    print("=" * 60)
    print("🚀 Open your browser and go to: http://localhost:8000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        with socketserver.TCPServer(("", port), VideoHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python server.py --port 8001")
        else:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual Video Blur Tool Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on (default: 8000)')
    
    args = parser.parse_args()
    start_server(args.port)