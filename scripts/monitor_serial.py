import serial
import time
import sys
port = '/dev/cu.usbserial-5AE90127161'
baud = 921600
timeout = 25
try:
    with serial.Serial(port, baud, timeout=0.1) as s:
        print(f"Listening on {port} (921600)...")
        # Try a soft reset first
        s.setDTR(False)
        s.setRTS(True)
        time.sleep(0.1)
        s.setRTS(False)
        time.sleep(0.1)
        
        start_time = time.time()
        found_data = False
        while time.time() - start_time < timeout:
            line = s.readline()
            if line:
                try:
                    decoded = line.decode('utf-8', errors='ignore').strip()
                    if decoded:
                        print(decoded)
                        sys.stdout.flush()
                        if "CSI_DATA" in decoded:
                            found_data = True
                except:
                    pass
        if not found_data:
            print("No CSI_DATA found within timeout.")
except Exception as e:
    print(f"Error: {e}")
