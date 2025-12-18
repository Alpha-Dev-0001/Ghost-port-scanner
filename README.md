# Network Scanner

This repository contains a basic Python-based network scanner designed for educational purposes. The script allows users to scan a specified IP address for open ports within a defined range, enabling better understanding of network security.

## Features

- **Single Host Scanning**: Scan a specific IP address for open ports.
- **Port Range Customization**: Define the range of ports to scan.
- **Multithreaded Scanning**: Speeds up the scan process by scanning multiple ports simultaneously.
- **Lightweight and Simple**: Easy-to-read and beginner-friendly Python code.

---

## Disclaimer
This script is intended for **educational use only**. Scanning devices or networks without proper authorization is illegal and unethical. Always obtain explicit permission before using this tool on any system.

---

## Prerequisites

Ensure the following before running the script:

- **Python 3.x** installed on your system.

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Alpha-Dev-0001/Ghost-port-scanner.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Ghost-port-scanner
   ```

3. Run the `network_scanner.py` script:
   ```bash
   python network_scanner.py
   ```

4. Enter the required inputs as prompted:
   - **Target Host**: IP address or hostname of the target machine (e.g., `192.168.1.1`).
   - **Starting Port**: The first port in the range to scan (e.g., `1`).
   - **Ending Port**: The last port in the range to scan (e.g., `1024`).

Example:

- To scan the target `192.168.1.1` for open ports between `1` and `1024`:
  ```bash
  python network_scanner.py
  ```
  Enter inputs as:
  ```
  Target Host: 192.168.1.1
  Starting Port: 1
  Ending Port: 1024
  ```

---

## Example Output

When the script finds open ports, it displays results like this:

```plaintext
[Scanning Host: 192.168.1.1] Ports 1-1024
[+] Host: 192.168.1.1, Port: 22 is OPEN
[+] Host: 192.168.1.1, Port: 80 is OPEN
[+] Host: 192.168.1.1, Port: 443 is OPEN
```

---

## Limitations and Notes

- **Accuracy**: The accuracy of the scan is affected by firewalls or network filters. Some open ports might appear closed if they're blocked by a firewall.
- **Scan Speed**: The script uses multithreading to improve scanning speed, but higher port ranges may still take time.
- **Permission Required**: Ensure you are legally authorized to scan the target host.

---

## Ethical Usage

- **Authorization**: Only use this tool on networks or devices that you own or have explicit permission to test.
- **Educational Purpose**: This tool is for learning and understanding port scanning techniques.
- **Responsibility**: The creator is not liable for any misuse of this tool.

---

## Contributing

If you'd like to contribute to the development or enhancement of this tool, feel free to submit a pull request or open an issue in the repository.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.