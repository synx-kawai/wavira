#!/usr/bin/env python3
"""
CSI Real-time Visualizer - Optimized PyQt6 + pyqtgraph
- Separate thread for serial reading
- Reduced update rate for smooth display
"""

import sys
import re
import threading
import numpy as np
from collections import deque
import serial
import serial.tools.list_ports

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
import pyqtgraph as pg

# Configuration
BAUD_RATE = 115200
HISTORY_SIZE = 80
UPDATE_INTERVAL = 100  # ms (10 Hz display update)


class SerialReader(QObject):
    """Separate thread for serial reading."""
    data_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.serial = None
        self.running = False
        self.thread = None
        self.last_emit = 0
        self.emit_interval = 0.08  # Throttle to ~12 Hz max

    def connect(self, port):
        self.disconnect()
        try:
            self.serial = serial.Serial(port, BAUD_RATE, timeout=0.1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Serial error: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.serial:
            try:
                self.serial.close()
            except:
                pass
            self.serial = None

    def _read_loop(self):
        """Background serial reading loop."""
        import time
        latest_data = None

        while self.running and self.serial:
            try:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("CSI_DATA"):
                        data = self._parse(line)
                        if data:
                            latest_data = data

                # Emit at throttled rate
                now = time.time()
                if latest_data and (now - self.last_emit) >= self.emit_interval:
                    self.data_received.emit(latest_data)
                    self.last_emit = now
                    latest_data = None
                else:
                    time.sleep(0.005)  # Small sleep to reduce CPU
            except Exception as e:
                if self.running:
                    print(f"Read error: {e}")
                break

    def _parse(self, line):
        """Parse CSI_DATA line."""
        try:
            match = re.search(r'\[([^\]]+)\]', line)
            if not match:
                return None

            csi_values = [int(x) for x in match.group(1).split(',')]
            parts = line.split(',')

            amplitudes = []
            for i in range(4, len(csi_values) - 1, 2):
                real = csi_values[i]
                imag = csi_values[i + 1]
                amplitudes.append((real**2 + imag**2) ** 0.5)

            return {
                'packet_num': int(parts[1]) if len(parts) > 1 else 0,
                'rssi': int(parts[3]) if len(parts) > 3 else 0,
                'channel': int(parts[4]) if len(parts) > 4 else 0,
                'amplitudes': amplitudes
            }
        except:
            return None


class CSIViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSI Viewer")
        self.setGeometry(100, 100, 900, 500)

        # Data
        self.amplitudes = []
        self.history = deque(maxlen=HISTORY_SIZE)
        self.packet_num = 0
        self.rssi = 0
        self.channel = 0

        # Serial reader
        self.reader = SerialReader()
        self.reader.data_received.connect(self.on_data)

        self.setup_ui()

        # Display update timer (separate from data receiving)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(UPDATE_INTERVAL)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)

        # Top bar
        top = QHBoxLayout()
        top.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.port_combo.currentTextChanged.connect(self.on_port_change)
        top.addWidget(self.port_combo)

        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red;")
        top.addWidget(self.status_label)

        self.info_label = QLabel("")
        top.addStretch()
        top.addWidget(self.info_label)
        layout.addLayout(top)

        # Plots - use OpenGL for GPU acceleration
        pg.setConfigOptions(antialias=False, useOpenGL=True)

        # Main plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#111')
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel('bottom', 'Subcarrier')
        self.plot.setLabel('left', 'Amplitude')
        self.curve = self.plot.plot(pen=pg.mkPen('#0f0', width=1))
        layout.addWidget(self.plot)

        # History plot
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setBackground('#111')
        self.hist_plot.showGrid(x=True, y=True, alpha=0.2)
        self.hist_plot.setMaximumHeight(120)
        self.hist_plot.setLabel('bottom', 'Time')
        self.hist_curve = self.hist_plot.plot(pen=pg.mkPen('#0f0', width=1))
        layout.addWidget(self.hist_plot)

    def refresh_ports(self):
        self.port_combo.clear()
        self.port_combo.addItem("-- Select --")
        for p in serial.tools.list_ports.comports():
            if 'usb' in p.device.lower():
                self.port_combo.addItem(p.device)

    def on_port_change(self, port):
        if port and port != "-- Select --":
            if self.reader.connect(port):
                self.status_label.setText("Connected")
                self.status_label.setStyleSheet("color: #0f0;")
            else:
                self.status_label.setText("Failed")
                self.status_label.setStyleSheet("color: red;")
        else:
            self.reader.disconnect()
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red;")

    def on_data(self, data):
        """Called when new CSI data arrives (from background thread)."""
        self.amplitudes = data['amplitudes']
        self.packet_num = data['packet_num']
        self.rssi = data['rssi']
        self.channel = data['channel']

        if self.amplitudes:
            self.history.append(np.mean(self.amplitudes))

    def update_display(self):
        """Update display at fixed rate (separate from data rate)."""
        if not self.amplitudes:
            return

        # Update plots
        self.curve.setData(self.amplitudes)

        if len(self.history) > 1:
            self.hist_curve.setData(list(self.history))

        # Update info
        self.info_label.setText(
            f"PKT:{self.packet_num} | RSSI:{self.rssi}dBm | CH:{self.channel} | Sub:{len(self.amplitudes)}"
        )

    def closeEvent(self, event):
        self.reader.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    p.setColor(QPalette.ColorRole.WindowText, QColor(0, 255, 0))
    p.setColor(QPalette.ColorRole.Base, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Text, QColor(0, 255, 0))
    p.setColor(QPalette.ColorRole.Button, QColor(40, 40, 40))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(0, 255, 0))
    app.setPalette(p)

    viewer = CSIViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
