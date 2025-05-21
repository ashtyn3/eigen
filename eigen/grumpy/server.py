import http

print("hi")


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        print("hi")


# if __name__ == "__main__":
print("listening on: http://localhost:8080")
h = http.server.HTTPServer(("localhost", 8080), Handler)
h.serve_forever()
