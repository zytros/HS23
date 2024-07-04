import json
import logging
import sys
import os
import socket
from schnorr import Schnorr, Schnorr_Params
from sage.all import QQ,ZZ,vector,matrix
import math
import numpy as np

# Change the port to match the challenge you're solving
PORT = 40200

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

def hidden_number_problem(msb, h, s):
    u = msb + 2**(127)-s
    return h,u

# Parameters of the P-256 NIST curve
a   = 0xffffffff00000001000000000000000000000000fffffffffffffffffffffffc
b   = 0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b
p   = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
P_x = 0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296
P_y = 0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5
q   = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551

nistp256_params = Schnorr_Params(a, b, p, P_x, P_y, q)
my_schnorr = Schnorr(nistp256_params)

sigs = []

for i in range(5):
    js = {"command": "get_signature",
        "msg": ("msg"+str(i))}
    json_send(js)
    msg_json = json_recv()
    sigs.append(msg_json)

ts = []
us = []
for sig in sigs:
    h = sig["h"]
    s = sig["s"]
    k = sig["nonce"]
    h,u = hidden_number_problem(k,h,s)
    ts.append(h)
    us.append(u)
w = vector(us+[0])


expon = 2**129
p_c = math.sqrt((2**254)*6) * expon / 2

M = matrix.identity(ZZ,5) * q * expon
M_b = matrix.block(ZZ,[[M,0], [matrix(ts) * expon,1]])
p_c_b = matrix.block(ZZ,[[M_b,0],[matrix(w) * expon,p_c]])

lll = p_c_b.LLL()
f = 0
i = 0
while type(f) == type(0):
    f_i = lll[i]
    if (f_i[-1] == p_c):
        f = f_i[:-1]/expon
        break
    i+=1

pks = []
v = w-f
sel = zip(ts,v)
for t,u in sel:
    t = my_schnorr.Z_q(t)
    u = my_schnorr.Z_q(u)
    t_inv = 1/t
    pks.append(t_inv * u)

for pk in pks:
    h,s = my_schnorr.Sign(pk,"gimme the flag")
    js = {"command": "solve",
          "h": int(h),
          "s": int(s)}
    json_send(js)
    msg_json = json_recv()
    if "flag" in msg_json:
        print(msg_json["flag"])
        break


'''messages = []
times = []
jsons = []
for i in range(1500):
    js = {"command": "get_signature"}
    json_send(js)
    msg_json = json_recv()
    jsons.append(msg_json)
    messages.append(msg_json["msg"])
    times.append(msg_json["time"])
part = np.argpartition(np.array(times),20)
six_leading = []
for i in range(20):
    six_leading.append(messages[part[i]])

js = {"command": "solve",
      "messages": six_leading}
json_send(js)
msg1_json = json_recv()
print(msg1_json["flag"])'''