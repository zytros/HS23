import json
import logging
import sys
import os
import socket
import ecdsa2 as ec

# Change the port to match the challenge you're solving
PORT = 40103

# Pro tip: for debugging, set the level to logging.DEBUG if you want
# to read all the messages back and forth from the server
# log_level = logging.DEBUG
log_level = logging.INFO
logging.basicConfig(stream=sys.stdout, level=log_level)

s = socket.socket()

# Set the environmental variable REMOTE to True in order to connect to the server
#
# To do so, run on the terminal:
# REMOTE=True sage solve.py
#
# When we grade, we will automatically set this for you
if "REMOTE" in os.environ:
    s.connect(("isl.aclabs.ethz.ch", PORT))
else:
    s.connect(("localhost", PORT))

fd = s.makefile("rw")


def json_recv():
    """Receive a serialized json object from the server and deserialize it"""

    line = fd.readline()
    logging.debug(f"Recv: {line}")
    return json.loads(line)

def json_send(obj):
    """Convert the object to json and send to the server"""

    request = json.dumps(obj)
    logging.debug(f"Send: {request}")
    fd.write(request + "\n")
    fd.flush()

# WRITE YOUR SOLUTION HERE

# A small test suite for verifying that everything works
a   = 0xffffffff00000001000000000000000000000000fffffffffffffffffffffffc
b   = 0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
p   = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
P_x = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296
P_y = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5
q   = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551

nistp256_params = ec.ECDSA2_Params(a, b, p, P_x, P_y, q)

my_ecdsa2 = ec.ECDSA2(nistp256_params)
#msg ="I can't not overthink it, it's impossible"
sk, pk = my_ecdsa2.KeyGen()
#r, s = my_ecdsa2.Sign(sk, msg)


x = pk.x
y = pk.y
js = {"command": "get_challenge",
      "x": int(x),
      "y": int(y)}
json_send(js)
msg_json = json_recv()
msg = msg_json["msg"]

(r,s) = my_ecdsa2.Sign(sk,msg)
js = {"command": "solve",
      "r": int(r),
      "s": int(s)}
json_send(js)
print(json_recv())
