
import sys
import socket
import tensorboard.program
import tensorboard.main

# Patch is_port_in_use to avoid "localhost" causing hangs on WSL2
# The original implementation uses "localhost" which can hang in some WSL2 configurations
def patched_is_port_in_use(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Check 127.0.0.1 explicitly instead of localhost
        return sock.connect_ex(('127.0.0.1', port)) == 0
    finally:
        sock.close()

print("Patching tensorboard.program.is_port_in_use for WSL2 compatibility...")
tensorboard.program.is_port_in_use = patched_is_port_in_use

if __name__ == "__main__":
    # run_main() parses sys.argv automatically
    sys.exit(tensorboard.main.run_main())
