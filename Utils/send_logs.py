# log_utils.py
import socket, json

mode = "no_page"
# mode = "page"

def debug_log(obj: dict, host='localhost', port=9999):
    if(mode == "page"):
        try:
            with socket.create_connection((host, port), timeout=1) as s:
                s.sendall((json.dumps(obj) + "\n").encode("utf-8"))
        except Exception:
            pass
    else:
        print(obj["message"])
        print("="*50)

