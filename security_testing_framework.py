# Educational Network Scanner Script
# Author: Alpha-Dev-0001
# Disclaimer: Use this script ONLY on networks and devices you own or have explicit permission to scan.

import socket
from concurrent.futures import ThreadPoolExecutor

def scan_host(host, port):
    """
    Scans a single port on a given host.
    """
    try:
        # Create a socket object
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Set timeout for a faster scan
            s.settimeout(1)
            # Try to connect to the host and port
            if s.connect_ex((host, port)) == 0:  # Connection success
                print(f"[+] Host: {host}, Port: {port} is OPEN")
    except Exception as e:
        print(f"[Error] Could not scan {host}:{port} - {e}")

def scan_network(host, start_port, end_port):
    """
    Scans a range of ports on a given host.
    """
    print(f"[Scanning Host: {host}] Ports {start_port}-{end_port}")
    
    # Use multithreading to speed up scanning
    with ThreadPoolExecutor(max_workers=100) as executor:
        for port in range(start_port, end_port + 1):
            executor.submit(scan_host, host, port)

def main():
    print("=== Network Scanner ===")
    host = input("Enter the target host (e.g., 192.168.1.1): ")
    start_port = int(input("Enter the starting port (e.g., 1): "))
    end_port = int(input("Enter the ending port (e.g., 1024): "))

    # Scan target
    scan_network(host, start_port, end_port)

if __name__ == "__main__":
    main()
