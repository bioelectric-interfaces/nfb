#!/usr/bin/env python

from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import json
from multiprocessing import Pool
import os

# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    # GET
    def do_GET(self):
        if self.path in ['/getlaststate']:

            # Send response status code
            self.send_response(200)

            # Send headers
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open("../../bci_current_state.pkl", "r", encoding="utf-8") as fp:
                state = float(fp.read())
            # Send message back to client
            state = 1 if state > 6000 else 2
            message = { "state": "{}".format(int(state)), "result": "true"}

            # Write content as utf-8 data
            print(json.dumps(message))
            self.wfile.write(bytes(json.dumps(message), "utf8"))
            #self.send_response(200, json.dumps(message))
        else:
            self.send_error(404)
        return


def run():
    print('starting server...')

    # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access

    #os.environ["CURRENT_BCI_STATE"] = "3"
    httpd = HTTPServer(('127.0.0.1', 336), testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()