#!/usr/bin/env python3
import socket
s = socket.socket()
s.settimeout(5)
try:
    r = s.connect_ex(("10.126.25.5", 40901))
    print("40901_reach", r)
except Exception as e:
    print("err", e)
finally:
    s.close()
