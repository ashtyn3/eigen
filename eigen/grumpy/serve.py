import os
from http import server
import json
import eigen
import base64
# from urllib import urlparse

kernel_data = b""


class Handler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        content = b""
        res_type = "text/html"
        if self.path == "/":
            print(os.path.join(os.path.dirname(__file__)))
            with open(
                os.path.join(os.path.dirname(__file__), "index.html"), "rb"
            ) as f:
                content = f.read()

        if self.path.startswith("/js/") or self.path.startswith("/vendor/"):
            with open(
                os.path.join(os.path.dirname(__file__), self.path.strip("/")),
                "rb",
            ) as f:
                content = f.read()
                res_type = "application/javascript"

        if self.path.startswith("/kernel"):
            content = kernel_data
            res_type = "application/json"

        self.send_response(200)
        self.send_header("Content-Type", res_type)
        self.send_header("Content-Length", len(content))
        self.end_headers()
        return self.wfile.write(content)


if __name__ == "__main__":
    print("listening on: http://localhost:8080")
    raw_data = bytes(os.getenv("G_DATA").encode())
    kernel_data = base64.b64decode(raw_data)
    h = server.HTTPServer(("localhost", 8080), Handler)
    h.serve_forever()
