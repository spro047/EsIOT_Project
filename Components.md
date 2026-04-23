# EsIOT Project - IoT Components List

This document outlines all the hardware and software components required to convert the AI-powered plant disease detection and crop recommendation system into a fully functional IoT project.

---

## 1. Sensing & Data Collection Components

### 1.1 Image Capture System
| Component | Specification | Purpose |
|-----------|---------------|---------|
| Raspberry Pi Camera Module 3 | 12MP, autofocus, 120° FOV | High-quality leaf image capture |
| USB Webcam (Alternative) | 1080p minimum | Backup camera option |
| Macro Lens Attachment | 10x-20x magnification | Close-up leaf detail capture |
| LED Ring Light | White 5600K, dimmable | Consistent lighting for images |
| Protective Housing | IP65 rated | Weatherproof camera enclosure |

### 1.2 Soil Sensors
| Component | Specification | Purpose |
|-----------|---------------|---------|
| NPK Soil Sensor | Nitrogen, Phosphorus, Potassium | Measure soil nutrient levels |
| Soil pH Sensor | Range 3-9 pH | Acidity/alkalinity measurement |
| Soil Moisture Sensor | Capacitive, 0-100% | Water content detection |
| Soil Temperature Sensor | -40°C to +80°C | Ground temperature monitoring |
| Soil EC Sensor | 0-20 mS/cm | Electrical conductivity |

### 1.3 Environmental Sensors
| Component | Specification | Purpose |
|-----------|---------------|---------|
| DHT22 Sensor | Temp: -40-80°C, Humidity: 0-100% | Ambient temperature & humidity |
| BMP280 Barometric Sensor | Pressure: 300-1100 hPa | Atmospheric pressure |
| Light Intensity Sensor (BH1750) | 1-65535 lux | Sunlight intensity |
| Rain Drop Sensor | Digital output | Rainfall detection |
| Anemometer | 0-100 m/s | Wind speed measurement |

---

## 2. Processing & Computing Components

### 2.1 Main Processing Unit
| Component | Specification | Purpose |
|-----------|---------------|---------|
| Raspberry Pi 5 | 8GB RAM, ARM Cortex-A76 | Primary edge computing unit |
| NVIDIA Jetson Nano | 128-core GPU | AI inference acceleration |
| ESP32-WROOM-32 | Dual-core, WiFi/BT | Secondary microcontroller |

### 2.2 Storage
| Component | Specification | Purpose |
|-----------|---------------|---------|
| microSD Card | 128GB, Class 10 | Local data storage |
| USB Flash Drive | 64GB | Backup storage |
| External SSD | 500GB, USB 3.0 | Large dataset storage |

### 2.3 Power Management
| Component | Specification | Purpose |
|-----------|---------------|---------|
| Power Supply | 5V 3A USB-C | Primary power |
| Solar Panel | 20W, 6V | Renewable energy |
| Battery Pack | 20000mAh, 5V | Backup power |
| Buck Converter | LM2596 module | Voltage regulation |
| Battery Management Module | TP4056 | Li-ion charging |

---

## 3. Connectivity & Communication Components

### 3.1 Wireless Communication
| Component | Specification | Purpose |
|-----------|---------------|---------|
| WiFi Module (Built-in) | 802.11 b/g/n | Cloud connectivity |
| LoRa Module | SX1278, 433MHz | Long-range farm communication |
| GSM/4G Module | SIM800L | Remote area connectivity |
| Bluetooth 5.0 | BLE 5.0 | Local device pairing |
| Zigbee Module | CC2530 | Sensor mesh network |

### 3.2 Network Equipment
| Component | Specification | Purpose |
|-----------|---------------|---------|
| WiFi Router | Dual-band, 2.4/5GHz | Local network |
| Ethernet Cable | Cat6, 10m | Wired connection |
| Network Switch | 8-port | Multiple device connectivity |
| Signal Booster | 4G/5G | Extended coverage |

---

## 4. Output & Display Components

### 4.1 User Interface
| Component | Specification | Purpose |
|-----------|---------------|---------|
| LCD Display | 16x2 I2C or 20x4 | Status display |
| OLED Display | 0.96", 128x64 | Graphical output |
| Touch Screen | 7", HDMI | Interactive dashboard |
| LED Indicators | RGB, 10mm | Status signals |
| Buzzer | 5V active | Audio alerts |

### 4.2 Alert System
| Component | Specification | Purpose |
|-----------|---------------|---------|
| Email Alerts | SMTP integration | Remote notifications |
| SMS Module | SIM800L | SMS alerts |
| Mobile App | Flutter/React Native | User dashboard |
| Telegram Bot | API integration | Instant messaging alerts |

---

## 5. Actuators & Control Components

### 5.1 Irrigation Control
| Component | Specification | Purpose |
|-----------|---------------|---------|
| Solenoid Valve | 12V, 1/2" N.C. | Water flow control |
| Water Pump | 12V, 5L/min | Water delivery |
| Relay Module | 8-channel, 5V coil | Device switching |
| Flow Meter | YF-S201 | Water usage tracking |

### 5.2 Spraying System
| Component | Specification | Purpose |
|-----------|---------------|---------|
| Mist Nozzle | 0.3mm orifice | Fine spray application |
| Peristaltic Pump | 12V, 100mL/min | Chemical delivery |
| Spray Tank | 10L capacity | Solution storage |

### 5.3 Motor Control
| Component | Specification | Purpose |
|-----------|---------------|------------------|
| Servo Motor | MG996R | Mechanical movement |
| Stepper Motor | 28BYJ-48 | Precision control |
| Motor Driver | L298N | Motor speed/direction |

---

## 6. Enclosure & Physical Components

### 6.1 Housing
| Component | Specification | Purpose |
|-----------|---------------|---------|
| IP66 Enclosure | 300x200x100mm | Main unit housing |
| Waterproof Box | ABS plastic | Sensor housing |
| Pole Mount | 50mm diameter | Installation pole |
| Solar Panel Mount | Adjustable angle | Panel positioning |

### 6.2 Cabling & Connectors
| Component | Specification | Purpose |
|-----------|---------------|---------|
| Jumper Wires | Male/Female set | Sensor connections |
| Dupont Cable | 20cm, 40-pin | Breadboard wiring |
| Power Cable | 2-pin, 1m | Power distribution |
| USB Cable | Type-C, 1m | Data/power |
| Cable Glands | M12, IP68 | Waterproof entry |

---

## 7. Software & Cloud Components

### 7.1 Edge Software
| Component | Purpose |
|-----------|---------|
| Python 3.10+ | Runtime environment |
| PyTorch (Optimized) | Model inference |
| OpenCV | Image preprocessing |
| Raspbian OS | Operating system |
| Docker | Containerization |

### 7.2 Cloud Platform
| Component | Purpose |
|-----------|---------|
| AWS IoT Core | Device management |
| MQTT Broker (Mosquitto) | Message queuing |
| InfluxDB | Time-series database |
| Grafana | Data visualization |
| Flask/FastAPI | REST API server |

### 7.3 Mobile Application
| Component | Purpose |
|-----------|---------|
| Flutter App | Cross-platform mobile app |
| Firebase | Authentication & storage |
| Push Notifications | Real-time alerts |

---

## 8. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EsIOT System Architecture                       │
└─────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────────┐
                            │   Cloud Server  │
                            │  (AWS/Firebase) │
                            │  - AI Model     │
                            │  - Database     │
                            │  - Dashboard    │
                            └────────┬────────┘
                                     │ Internet
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌────────▼────────┐
    │   Field Node 1    │  │   Field Node 2    │  │   Field Node N  │
    │  (Raspberry Pi)   │  │  (Raspberry Pi)   │  │  (Raspberry Pi) │
    │  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────┐│
    │  │ Camera      │  │  │  │ Camera      │  │  │  │ Camera      ││
    │  │ + LED Light │  │  │  │ + LED Light │  │  │  │ + LED Light ││
    │  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────┘│
    │  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────┐│
    │  │ Soil Sensors│  │  │  │ Soil Sensors│  │  │  │ Soil Sensors││
    │  │ NPK/pH/Moist│  │  │  │ NPK/pH/Moist│  │  │  │ NPK/pH/Moist││
    │  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────┘│
    │  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────┐│
    │  │ Env Sensors │  │  │  │ Env Sensors │  │  │  │ Env Sensors ││
    │  │ DHT22/BMP280│  │  │  │ DHT22/BMP280│  │  │  │ DHT22/BMP280││
    │  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────┘│
    │  ┌─────────────┐  │  │  ┌─────────────┐  │  │  ┌─────────────┐│
    │  │ Actuators   │  │  │  │ Actuators   │  │  │  │ Actuators   ││
    │  │ Relay/Pump  │  │  │  │ Relay/Pump  │  │  │  │ Relay/Pump  ││
    │  └─────────────┘  │  │  └─────────────┘  │  │  └─────────────┘│
    └─────────┬─────────┘  └─────────┬─────────┘  └────────┬─────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                            ┌────────▼────────┐
                            │   LoRa Gateway  │
                            │   (ESP32)       │
                            └─────────────────┘
```

---

## 9. Estimated Cost Breakdown

| Category | Components | Estimated Cost (USD) |
|----------|------------|---------------------|
| Processing | Raspberry Pi 5, ESP32 | $80 - $100 |
| Camera & Optics | Camera Module 3, Lens, LED | $40 - $60 |
| Soil Sensors | NPK, pH, Moisture, Temp | $30 - $50 |
| Environmental | DHT22, BMP280, Light Sensor | $15 - $25 |
| Connectivity | LoRa, GSM modules | $25 - $40 |
| Power | Solar panel, Battery, Buck converter | $40 - $60 |
| Display | LCD/OLED, Touch screen | $15 - $35 |
| Actuators | Relay, Pump, Valves | $30 - $50 |
| Enclosure | IP66 box, Mounts, Cables | $25 - $40 |
| Cloud/Software | AWS Free Tier, Firebase | $0 - $20 |
| **Total** | | **$300 - $480** |

---

## 10. Implementation Phases

### Phase 1: Basic Setup
- [ ] Raspberry Pi setup with camera
- [ ] Basic image capture system
- [ ] Local AI model deployment

### Phase 2: Sensor Integration
- [ ] Soil sensor calibration
- [ ] Environmental sensor wiring
- [ ] Data logging system

### Phase 3: Connectivity
- [ ] WiFi/LoRa setup
- [ ] Cloud connection
- [ ] Mobile app development

### Phase 4: Automation
- [ ] Relay and actuator control
- [ ] Automated irrigation
- [ ] Alert system

### Phase 5: Optimization
- [ ] Power optimization (solar)
- [ ] Model optimization for edge
- [ ] Full system testing

---

## 11. References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [ESP32 Documentation](https://docs.espressif.com/)
- [AWS IoT Core](https://aws.amazon.com/iot-core/)
- [MQTT Protocol](http://mqtt.org/)

---

*Document Version: 1.0*  
*Last Updated: April 2026*