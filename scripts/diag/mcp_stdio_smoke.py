import json
import queue
import subprocess
import sys
import threading

sys.stdout.reconfigure(encoding="utf-8")

PY = r"D:\MCP_1С_homework\.venv\Scripts\python.exe"
CWD = r"D:\MCP_1С_homework"
ERRLOG = r"C:\Users\smatu\AppData\Local\Temp\mcp_stdio_stderr.log"

err = open(ERRLOG, "w", encoding="utf-8")
proc = subprocess.Popen(
    [PY, "-m", "mcp_1c_rag"],
    cwd=CWD,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=err,
    text=True,
    encoding="utf-8",
)

q = queue.Queue()


def reader():
    for line in proc.stdout:
        q.put(line.rstrip("\n"))


threading.Thread(target=reader, daemon=True).start()


def send(obj):
    proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def recv(timeout):
    try:
        return json.loads(q.get(timeout=timeout))
    except queue.Empty:
        return None


def rpc(id_, method, params=None, timeout=60):
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    send(msg)
    while True:
        resp = recv(timeout)
        if resp is None:
            return None
        if resp.get("id") == id_:
            return resp
        # skip notifications/other messages


# 1. initialize
resp = rpc(
    1,
    "initialize",
    {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "zcode-test", "version": "1.0"},
    },
    timeout=90,
)
if resp is None:
    print("FAIL: initialize TIMEOUT (server did not answer in 90s)")
else:
    print("INITIALIZE OK:", json.dumps(resp.get("result", resp), ensure_ascii=False)[:400])

# 2. initialized notification
send({"jsonrpc": "2.0", "method": "notifications/initialized"})

# 3. tools/list
resp = rpc(2, "tools/list", {}, timeout=60)
if resp is None:
    print("FAIL: tools/list TIMEOUT")
else:
    tools = [t.get("name") for t in resp.get("result", {}).get("tools", [])]
    print("TOOLS/LIST OK:", tools)

# 4. tools/call codesearch
resp = rpc(
    3,
    "tools/call",
    {
        "name": "codesearch",
        "arguments": {"query": "начисление НДС с аванса", "limit": 5},
    },
    timeout=300,
)
if resp is None:
    print("FAIL: tools/call TIMEOUT (300s)")
elif "error" in resp:
    print("TOOLS/CALL ERROR:", json.dumps(resp["error"], ensure_ascii=False)[:2000])
else:
    content = resp.get("result", {}).get("content", [])
    for c in content:
        if c.get("type") == "text":
            print("TOOLS/CALL codesearch RESULT:")
            print(c.get("text", ""))

# 5. tools/call metadatasearch
resp = rpc(
    4,
    "tools/call",
    {
        "name": "metadatasearch",
        "arguments": {"query": "начисление НДС с аванса счет-фактура", "limit": 5},
    },
    timeout=300,
)
if resp is None:
    print("FAIL: metadatasearch TIMEOUT (300s)")
elif "error" in resp:
    print("TOOLS/CALL ERROR:", json.dumps(resp["error"], ensure_ascii=False)[:2000])
else:
    content = resp.get("result", {}).get("content", [])
    for c in content:
        if c.get("type") == "text":
            print("TOOLS/CALL metadatasearch RESULT:")
            print(c.get("text", ""))

proc.terminate()
err.close()
print("--- stderr tail ---")
with open(ERRLOG, encoding="utf-8") as f:
    tail = f.read()[-1500:]
print(tail)
