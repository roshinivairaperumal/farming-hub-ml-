from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import json, io, numpy as np, tensorflow as tf, requests
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64
import time

app = Flask(__name__)
CORS(app)

MODEL_KERAS = "disease_model.keras"
MODEL_H5 = "disease_model.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = (256, 256)
ESP32_CAM_URL = "http://10.37.1.217"  # UPDATED IP ADDRESS

DISEASE_INFO = {
    "healthy": {
        "display_name": "🌱 Healthy Plant",
        "severity": "None",
        "description": "The leaf shows no signs of disease or stress. Color, texture, and structure appear normal and vibrant.",
        "prevention": [
            "Maintain consistent watering (1-2 inches per week)",
            "Ensure 6-8 hours of sunlight daily",
            "Apply balanced NPK fertilizer (10-10-10) every 4-6 weeks",
            "Conduct weekly visual inspections for early pest detection",
            "Maintain soil pH between 6.0-7.0 for optimal nutrient uptake"
        ],
        "treatment": "Continue current maintenance routine. No chemical treatment required.",
        "immediate_actions": ["Continue monitoring", "Maintain watering schedule"],
        "color": "green",
        "confidence_threshold": 0.85
    },
    "powdery_mildew": {
        "display_name": "🍄 Powdery Mildew",
        "severity": "Moderate",
        "description": "Characterized by white, powdery fungal growth on upper leaf surfaces. Can cause leaf curling, yellowing, and premature leaf drop. Thrives in temperatures 60-80°F with high humidity.",
        "prevention": [
            "Space plants 2-3 feet apart for proper air circulation",
            "Water at soil level (avoid overhead watering)",
            "Apply sulfur-based preventative spray every 10-14 days in humid conditions",
            "Remove and destroy infected plant debris in fall",
            "Plant resistant varieties like 'Immunox'-treated seeds"
        ],
        "treatment": "Apply fungicide containing myclobutanil or potassium bicarbonate. Spray every 7-10 days until controlled. For organic treatment: 1 tbsp baking soda + 1 tsp horticultural oil + 1 gallon water.",
        "immediate_actions": [
            "Remove severely infected leaves",
            "Apply fungicide within 48 hours",
            "Improve air circulation around plant",
            "Avoid nitrogen-heavy fertilizers"
        ],
        "color": "blue",
        "confidence_threshold": 0.75
    },
    "rust": {
        "display_name": "🦠 Rust Disease",
        "severity": "Moderate to High",
        "description": "Identified by orange-brown pustules on leaf undersides. Yellow spots appear on upper surfaces. Spreads rapidly in moist conditions (70-80°F). Can defoliate plants if untreated.",
        "prevention": [
            "Apply copper fungicide early in growing season",
            "Water in morning to allow leaves to dry",
            "Remove weed hosts (especially wild berries)",
            "Ensure 4-6 hours of morning sun for quick drying",
            "Use drip irrigation instead of sprinklers"
        ],
        "treatment": "Apply systemic fungicide containing azoxystrobin or chlorothalonil. Treat every 7-14 days during humid periods. Remove and bag severely infected leaves.",
        "immediate_actions": [
            "Isolate infected plants immediately",
            "Apply fungicide within 24-48 hours",
            "Remove bottom 1/3 of affected leaves",
            "Disinfect tools with 10% bleach solution"
        ],
        "color": "orange",
        "confidence_threshold": 0.70
    }
}


try:
    model = tf.keras.models.load_model(MODEL_KERAS)
    print("✅ Loaded .keras model")
except Exception:
    try:
        model = tf.keras.models.load_model(MODEL_H5)
        print("✅ Loaded .h5 model")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None

with open(LABELS_PATH) as f:
    class_idx = json.load(f)["class_indices"]
idx2name = {v: k for k, v in class_idx.items()}

print("🌿 ML Model loaded and ready!")

def prepare(img: Image.Image, target_size=IMG_SIZE):
    img = img.convert("RGB").resize(target_size)
    x = np.asarray(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)
    return x

def flip_image(img: Image.Image):
    """Flip image horizontally to correct mirror effect"""
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def capture_from_esp32():
    """Capture image from ESP32-CAM - optimized for CameraWebServer"""
    base_url = ESP32_CAM_URL  # USING UPDATED IP
    
    # Different endpoints that various ESP32-CAM firmwares use
    endpoints = [
        '/capture',           # Common endpoint
        '/jpg',               # Another common endpoint
        '/snapshot',          # ESP32-CAM snapshot
        '/',                  # Root might serve image
        '/cam-lo.jpg',        # Low-res stream
        '/cam-hi.jpg',        # High-res stream
        '/photo.jpg',         # Photo endpoint
        '/image.jpg'          # Image endpoint
    ]
    
    print(f"🔄 Attempting to capture from ESP32-CAM at {base_url}")
    
    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            print(f"   🔍 Trying endpoint: {url}")
            
            # Add a timestamp to avoid cached images
            timestamp_url = f"{url}?t={int(time.time()*1000)}"
            
            response = requests.get(timestamp_url, timeout=10)
            print(f"   📊 Response status: {response.status_code}")
            print(f"   📦 Content length: {len(response.content)} bytes")
            print(f"   🎯 Content type: {response.headers.get('content-type', 'unknown')}")
            
            # Check if we got a valid image
            if response.status_code == 200 and len(response.content) > 1000:
                # Try to open as image to verify
                try:
                    img = Image.open(io.BytesIO(response.content))
                    # Verify it's a valid image by checking size
                    img.verify()
                    # Reopen after verify
                    img = Image.open(io.BytesIO(response.content))
                    
                    print(f"✅ SUCCESS with image endpoint: {endpoint}")
                    print(f"✅ Image dimensions: {img.size}")
                    
                    # Flip image to correct orientation
                    img = flip_image(img)
                    print("✅ Image flipped to correct orientation")
                    
                    return img
                except Exception as e:
                    print(f"   ❌ Received data but not a valid image: {e}")
                    continue
            else:
                print(f"❌ Endpoint {endpoint} failed - Status: {response.status_code}, Size: {len(response.content)}")
                    
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection error - ESP32 not reachable at {url}")
            continue
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout - ESP32 not responding at {url}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Endpoint {endpoint} failed: {e}")
            continue
        except Exception as e:
            print(f"   ❌ Endpoint {endpoint} error: {e}")
            continue
    
    print("❌ No valid image endpoints found")
    return None

def get_disease_info(disease_name):
    """Get disease information with intelligent matching"""
    disease_lower = disease_name.lower()
    
    name_mapping = {
        'powdery': 'powdery_mildew',
        'powdery_mold': 'powdery_mildew', 
        'white_mold': 'powdery_mildew',
        'rust': 'rust',
        'leaf_rust': 'rust',
        'healthy': 'healthy',
        'normal': 'healthy'
    }
    
    if disease_lower in DISEASE_INFO:
        return DISEASE_INFO[disease_lower]
    
    for key, standard_name in name_mapping.items():
        if key in disease_lower:
            return DISEASE_INFO[standard_name]
    
    return {
        "display_name": f"🔍 {disease_name.replace('_', ' ').title()}",
        "severity": "Requires Expert Diagnosis",
        "description": "This condition requires professional diagnosis. Symptoms don't match common diseases in our database.",
        "prevention": [
            "Isolate affected plant immediately",
            "Take clear photos from multiple angles",
            "Contact local agricultural extension office"
        ],
        "treatment": "Consultation with plant pathologist recommended.",
        "immediate_actions": ["Isolate plant", "Document symptoms", "Seek expert advice"],
        "color": "gray",
        "confidence_threshold": 0.60
    }

def get_response_time(severity):
    response_times = {
        "Critical": "IMMEDIATE (Within 12 hours)",
        "High": "URGENT (Within 24 hours)", 
        "Moderate to High": "PRIORITY (Within 48 hours)",
        "Moderate": "TIMELY (Within 72 hours)",
        "Low to Moderate": "SCHEDULED (Within 1 week)",
        "Low": "MONITOR (Regular observation)",
        "None": "ROUTINE (Continue maintenance)"
    }
    return response_times.get(severity, "Within 48 hours")


@app.route('/test-esp32')
def test_esp32():
    """Test ESP32-CAM connection directly with enhanced debugging"""
    base_url = ESP32_CAM_URL  # USING UPDATED IP
    endpoints = ['/capture', '/jpg', '/snapshot', '/photo', '/image', '/cam.jpg', '/', '/cam-lo.jpg', '/cam-hi.jpg']
    
    results = {}
    
    # First check if ESP32 is reachable at all
    try:
        ping_response = requests.get(base_url, timeout=3)
        results['base_url_reachable'] = {
            "status_code": ping_response.status_code,
            "reachable": True
        }
    except Exception as e:
        results['base_url_reachable'] = {
            "reachable": False,
            "error": str(e)
        }
    
    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}?t={int(time.time()*1000)}"
            response = requests.get(url, timeout=5)
            
            # Check if it's actually an image
            is_image = False
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type or response.content[:2] in [b'\xff\xd8', b'\x89\x50']:
                is_image = True
            
            results[endpoint] = {
                "status_code": response.status_code,
                "content_type": content_type,
                "content_length": len(response.content),
                "is_image": is_image,
                "success": response.status_code == 200 and len(response.content) > 1000 and is_image
            }
        except Exception as e:
            results[endpoint] = {"error": str(e), "success": False}
    
    working_endpoints = []
    for ep, result in results.items():
        if isinstance(result, dict) and result.get('success', False):
            working_endpoints.append(ep)
    
    return jsonify({
        "esp32_url": base_url,
        "test_results": results,
        "working_endpoints": working_endpoints,
        "recommended_endpoint": working_endpoints[0] if working_endpoints else None
    })


@app.route('/')
def home():
    """Main web interface"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AgriSense - Plant Disease Detection</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --primary: #2E7D32;
                --secondary: #4CAF50;
                --dark: #1B5E20;
                --light: #E8F5E9;
                --warning: #FF9800;
                --danger: #f44336;
            }
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            body {
                background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
                color: #333;
                line-height: 1.6;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background: linear-gradient(135deg, var(--primary), var(--dark));
                color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                flex-wrap: wrap;
                background: white;
                padding: 10px;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }
            .tab-button {
                padding: 12px 24px;
                background: white;
                border: 2px solid var(--primary);
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s;
                flex: 1;
                min-width: 150px;
                text-align: center;
            }
            .tab-button.active {
                background: var(--primary);
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
            }
            .tab-content {
                display: none;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .tab-content.active {
                display: block;
                animation: fadeIn 0.5s ease-in;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .upload-area {
                border: 2px dashed var(--primary);
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                cursor: pointer;
                transition: all 0.3s;
                background: var(--light);
            }
            .upload-area:hover {
                background: #e1f5e1;
                transform: translateY(-2px);
            }
            button {
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s;
                margin: 10px 5px;
            }
            button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none !important;
            }
            .result-section {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                display: none;
            }
            .disease-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                border-left: 4px solid var(--primary);
            }
            .confidence-bar {
                height: 10px;
                background: #e0e0e0;
                border-radius: 5px;
                margin: 10px 0;
                overflow: hidden;
            }
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--primary), var(--secondary));
                border-radius: 5px;
                transition: width 0.5s;
            }
            .stream-container {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                margin: 20px 0;
                background: #000;
                text-align: center;
                padding: 10px;
                position: relative;
            }
            .stream-container img {
                max-width: 100%;
                max-height: 400px;
                border-radius: 8px;
                transition: transform 0.3s;
                transform: scaleX(-1); /* Flip ESP32 stream horizontally */
            }
            .stream-container img:hover {
                transform: scaleX(-1) scale(1.02); /* Maintain flip on hover */
            }
            .capture-overlay {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 12px;
            }
            .status-indicator {
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
                display: inline-block;
                margin: 10px 0;
            }
            .status-connected {
                background: #E8F5E9;
                color: var(--dark);
                border: 1px solid var(--primary);
            }
            .status-disconnected {
                background: #FFEBEE;
                color: var(--danger);
                border: 1px solid var(--danger);
            }
            .preview-image {
                max-width: 100%;
                max-height: 300px;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);
                margin: 10px 0;
                border: 3px solid var(--primary);
            }
            .device-camera-preview {
                max-width: 100%;
                max-height: 300px;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);
                margin: 10px 0;
                border: 3px solid var(--primary);
                transform: scaleX(-1); /* Flip device camera preview */
            }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .video-controls {
                display: flex;
                justify-content: center;
                gap: 10px;
                margin: 15px 0;
                flex-wrap: wrap;
            }
            .video-controls button {
                flex: 1;
                min-width: 120px;
            }
            .live-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                background: #f44336;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.4; }
                100% { opacity: 1; }
            }
            .camera-controls {
                display: flex;
                justify-content: center;
                gap: 10px;
                margin: 15px 0;
                flex-wrap: wrap;
            }
            .debug-info {
                background: #f0f0f0;
                padding: 15px;
                border-radius: 8px;
                font-family: monospace;
                font-size: 12px;
                margin: 10px 0;
                text-align: left;
                max-height: 300px;
                overflow: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-seedling"></i> AgriSense - Plant Disease Detection</h1>
                <p>AI-Powered Plant Health Analysis with Multiple Capture Options</p>
            </div>

            <div class="tabs">
                <button class="tab-button active" onclick="openTab('upload')">
                    <i class="fas fa-upload"></i> Upload Image
                </button>
                <button class="tab-button" onclick="openTab('camera')">
                    <i class="fas fa-camera"></i> Device Camera
                </button>
                <button class="tab-button" onclick="openTab('esp32')">
                    <i class="fas fa-satellite-dish"></i> ESP32-CAM
                </button>
                <button class="tab-button" onclick="openTab('debug')" style="background: var(--warning);">
                    <i class="fas fa-bug"></i> Debug
                </button>
            </div>

            <!-- Upload Tab -->
            <div id="upload" class="tab-content active">
                <h2><i class="fas fa-upload"></i> Upload Image Analysis</h2>
                <p>Upload a clear image of a plant leaf for disease detection</p>
                
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <i class="fas fa-cloud-upload-alt" style="font-size: 48px; color: var(--primary); margin-bottom: 15px;"></i>
                    <h3>Click to Upload Image</h3>
                    <p>Supported formats: JPG, JPEG, PNG</p>
                    <input type="file" id="fileInput" accept="image/*" style="display: none" onchange="handleFileUpload(this.files[0])">
                </div>
                
                <div id="uploadPreview" style="text-align: center; margin: 20px 0;"></div>
                
                <button onclick="analyzeUpload()" id="analyzeUploadBtn" style="display: none; width: 100%;">
                    <i class="fas fa-microscope"></i> Analyze Image
                </button>
            </div>

            <!-- Camera Tab -->
            <div id="camera" class="tab-content">
                <h2><i class="fas fa-camera"></i> Device Camera Analysis</h2>
                <p>Use your device camera to capture and analyze plant leaves in real-time</p>
                
                <div id="cameraContainer" style="text-align: center;">
                    <div id="cameraPlaceholder" style="display: block;">
                        <p>Camera will start when you click the button below</p>
                    </div>
                    <video id="video" width="100%" style="border-radius: 10px; display: none; max-height: 400px; transform: scaleX(-1);"></video>
                    <canvas id="canvas" style="display: none;"></canvas>
                    
                    <div class="camera-controls">
                        <button onclick="startCamera()" id="startCameraBtn">
                            <i class="fas fa-camera"></i> Start Camera
                        </button>
                        <button onclick="captureImage()" id="captureBtn" style="display: none;">
                            <i class="fas fa-camera-retro"></i> Capture & Analyze
                        </button>
                        <button onclick="stopCamera()" id="stopCameraBtn" style="display: none; background: var(--danger);">
                            <i class="fas fa-stop"></i> Stop Camera
                        </button>
                    </div>
                </div>
                
                <div id="cameraPreview" style="text-align: center; margin: 20px 0;"></div>
            </div>

            <!-- ESP32-CAM Tab -->
            <div id="esp32" class="tab-content">
                <h2><i class="fas fa-satellite-dish"></i> ESP32-CAM Live Monitoring</h2>
                <p>Connect to field camera for real-time plant monitoring and analysis</p>
                
                <div id="esp32Status" style="text-align: center; margin: 20px 0;">
                    <div class="status-indicator status-disconnected" id="connectionStatus">
                        <i class="fas fa-unlink"></i> Checking ESP32-CAM Connection...
                    </div>
                </div>
                
                <div class="stream-container">
                    <div class="capture-overlay">
                        <span class="live-indicator"></span>LIVE
                    </div>
                    <img id="esp32Stream" src="" alt="ESP32-CAM Stream" 
                         onload="updateConnectionStatus(true)"
                         onerror="updateConnectionStatus(false)">
                </div>
                
                <div class="video-controls">
                    <button onclick="captureFromESP32()" id="esp32CaptureBtn">
                        <i class="fas fa-camera"></i> Capture & Analyze
                    </button>
                    <button onclick="refreshESP32Stream()" style="background: var(--warning);">
                        <i class="fas fa-sync-alt"></i> Refresh Stream
                    </button>
                    <button onclick="testESP32Connection()" style="background: var(--primary);">
                        <i class="fas fa-search"></i> Test Connection
                    </button>
                </div>
                
                <div id="esp32Preview" style="text-align: center; margin: 20px 0;"></div>
                <div id="esp32DebugInfo" class="debug-info" style="display: none;"></div>
            </div>

            <!-- Debug Tab -->
            <div id="debug" class="tab-content">
                <h2><i class="fas fa-bug"></i> Debug ESP32-CAM Connection</h2>
                <p>Test and troubleshoot the ESP32-CAM connection</p>
                
                <div style="text-align: center; margin: 20px 0;">
                    <button onclick="runFullDiagnostic()" style="background: var(--warning); width: 100%;">
                        <i class="fas fa-stethoscope"></i> Run Full Diagnostic
                    </button>
                </div>
                
                <div id="debugResults" style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;"></div>
            </div>

            <!-- Results Section -->
            <div id="resultSection" class="result-section">
                <h2><i class="fas fa-chart-line"></i> Analysis Results</h2>
                <div id="resultsContent"></div>
            </div>
        </div>

        <script>
            const API_BASE = window.location.origin;
            const ESP32_URL = "http://10.37.1.217";  // UPDATED IP ADDRESS

            let currentImage = null;
            let stream = null;
            let esp32StreamInterval = null;
            let isESP32Streaming = false;

            function updateConnectionStatus(isConnected) {
                const statusEl = document.getElementById('connectionStatus');
                if (isConnected) {
                    statusEl.className = 'status-indicator status-connected';
                    statusEl.innerHTML = '<i class="fas fa-link"></i> ESP32-CAM Connected';
                } else {
                    statusEl.className = 'status-indicator status-disconnected';
                    statusEl.innerHTML = '<i class="fas fa-unlink"></i> ESP32-CAM Disconnected';
                }
            }

            // Initialize ESP32 stream
            function initializeESP32Stream() {
                const streamImg = document.getElementById('esp32Stream');
                // Try different endpoints for the stream
                const endpoints = ['/capture', '/jpg', '/snapshot', '/cam-lo.jpg'];
                // Use the first one that works
                streamImg.src = ESP32_URL + '/capture?t=' + Date.now();
            }

            function openTab(tabName) {
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                // Remove active class from all buttons
                document.querySelectorAll('.tab-button').forEach(button => {
                    button.classList.remove('active');
                });
                // Show current tab
                document.getElementById(tabName).classList.add('active');
                event.currentTarget.classList.add('active');
                
                // Stop camera when switching tabs
                if (tabName !== 'camera' && stream) {
                    stopCamera();
                }
                
                // Initialize ESP32 stream when switching to that tab
                if (tabName === 'esp32') {
                    initializeESP32Stream();
                    if (!isESP32Streaming) {
                        startESP32Stream();
                    }
                } else {
                    stopESP32Stream();
                }
            }

            function startESP32Stream() {
                if (isESP32Streaming) return;
                
                isESP32Streaming = true;
                
                // Refresh image every 2 seconds to simulate video stream
                esp32StreamInterval = setInterval(() => {
                    if (document.getElementById('esp32').classList.contains('active')) {
                        const streamImg = document.getElementById('esp32Stream');
                        streamImg.src = ESP32_URL + '/capture?t=' + Date.now();
                    }
                }, 2000);
                
                console.log('ESP32-CAM stream started');
            }

            function stopESP32Stream() {
                isESP32Streaming = false;
                
                if (esp32StreamInterval) {
                    clearInterval(esp32StreamInterval);
                    esp32StreamInterval = null;
                }
                
                console.log('ESP32-CAM stream stopped');
            }

            function handleFileUpload(file) {
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        currentImage = file;
                        document.getElementById('uploadPreview').innerHTML = `
                            <img src="${e.target.result}" class="preview-image">
                            <p><i class="fas fa-check-circle" style="color: var(--primary);"></i> Image ready for analysis</p>
                        `;
                        document.getElementById('analyzeUploadBtn').style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            }

            async function analyzeUpload() {
                if (!currentImage) {
                    alert('Please select an image first');
                    return;
                }
                
                const btn = document.getElementById('analyzeUploadBtn');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<div class="loading"></div> Analyzing...';
                btn.disabled = true;

                const formData = new FormData();
                formData.append('file', currentImage);

                try {
                    const response = await fetch(API_BASE + '/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    alert('Error analyzing image: ' + error.message);
                } finally {
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                }
            }

            // Camera functions
            async function startCamera() {
                try {
                    document.getElementById('cameraPlaceholder').style.display = 'none';
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            width: { ideal: 1280 },
                            height: { ideal: 720 },
                            facingMode: 'environment'
                        } 
                    });
                    const video = document.getElementById('video');
                    video.srcObject = stream;
                    video.style.display = 'block';
                    video.play();
                    
                    document.getElementById('startCameraBtn').style.display = 'none';
                    document.getElementById('captureBtn').style.display = 'inline-block';
                    document.getElementById('stopCameraBtn').style.display = 'inline-block';
                    
                } catch (error) {
                    alert('Error accessing camera: ' + error.message);
                    console.error('Camera error:', error);
                }
            }

            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
                document.getElementById('video').style.display = 'none';
                document.getElementById('video').srcObject = null;
                document.getElementById('startCameraBtn').style.display = 'inline-block';
                document.getElementById('captureBtn').style.display = 'none';
                document.getElementById('stopCameraBtn').style.display = 'none';
                document.getElementById('cameraPlaceholder').style.display = 'block';
                document.getElementById('cameraPreview').innerHTML = '';
            }

            async function captureImage() {
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                const context = canvas.getContext('2d');
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                // Flip the canvas context to correct mirror effect
                context.translate(canvas.width, 0);
                context.scale(-1, 1);
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Reset transformation
                context.setTransform(1, 0, 0, 1, 0, 0);
                
                // Show loading state
                document.getElementById('cameraPreview').innerHTML = '<div class="loading"></div> Processing image...';
                
                canvas.toBlob(async function(blob) {
                    currentImage = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
                    
                    // Show preview (flipped for consistency)
                    const previewUrl = URL.createObjectURL(blob);
                    document.getElementById('cameraPreview').innerHTML = `
                        <img src="${previewUrl}" class="device-camera-preview">
                        <p><i class="fas fa-check-circle" style="color: var(--primary);"></i> Image captured</p>
                        <div class="loading"></div> Analyzing...
                    `;
                    
                    // Auto-analyze
                    const formData = new FormData();
                    formData.append('file', currentImage);

                    try {
                        const response = await fetch(API_BASE + '/predict', {
                            method: 'POST',
                            body: formData
                        });
                        const result = await response.json();
                        displayResults(result);
                    } catch (error) {
                        document.getElementById('cameraPreview').innerHTML += `<p style="color: var(--danger);">Error: ${error.message}</p>`;
                    }
                }, 'image/jpeg');
            }

            async function captureFromESP32() {
                const btn = document.getElementById('esp32CaptureBtn');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<div class="loading"></div> Capturing...';
                btn.disabled = true;

                document.getElementById('esp32Preview').innerHTML = '<div class="loading"></div> Capturing image from ESP32-CAM...';
                document.getElementById('esp32DebugInfo').style.display = 'none';

                try {
                    console.log('Sending capture request to:', API_BASE + '/capture-from-esp32');
                    const response = await fetch(API_BASE + '/capture-from-esp32');
                    const result = await response.json();
                    
                    console.log('Capture response:', result);
                    
                    if (result.status === 'success') {
                        document.getElementById('esp32Preview').innerHTML = `
                            <img src="data:image/jpeg;base64,${result.image_data}" class="preview-image">
                            <p><i class="fas fa-check-circle" style="color: var(--primary);"></i> Image captured from ESP32-CAM</p>
                            <p><strong>Disease:</strong> ${result.prediction}</p>
                            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        `;
                        displayResults(result);
                    } else {
                        document.getElementById('esp32Preview').innerHTML = `
                            <p style="color: var(--danger);"><i class="fas fa-exclamation-triangle"></i> Error: ${result.error}</p>
                            <button onclick="testESP32Connection()" style="margin-top: 10px;">
                                <i class="fas fa-search"></i> Test Connection
                            </button>
                        `;
                    }
                } catch (error) {
                    console.error('Capture error:', error);
                    document.getElementById('esp32Preview').innerHTML = `
                        <p style="color: var(--danger);"><i class="fas fa-exclamation-triangle"></i> Network Error: ${error.message}</p>
                        <button onclick="testESP32Connection()" style="margin-top: 10px;">
                            <i class="fas fa-search"></i> Test Connection
                        </button>
                    `;
                } finally {
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                }
            }

            function refreshESP32Stream() {
                const streamImg = document.getElementById('esp32Stream');
                streamImg.src = ESP32_URL + '/capture?t=' + Date.now();
                document.getElementById('esp32Preview').innerHTML = '<p>Refreshing stream...</p>';
                setTimeout(() => {
                    document.getElementById('esp32Preview').innerHTML = '';
                }, 2000);
            }

            async function testESP32Connection() {
                document.getElementById('esp32Preview').innerHTML = '<div class="loading"></div> Testing ESP32-CAM connection...';
                document.getElementById('esp32DebugInfo').style.display = 'block';
                
                try {
                    const response = await fetch(API_BASE + '/test-esp32');
                    const result = await response.json();
                    
                    let html = '<h4>ESP32-CAM Connection Test Results</h4>';
                    html += `<p><strong>ESP32 URL:</strong> ${result.esp32_url}</p>`;
                    
                    if (result.working_endpoints && result.working_endpoints.length > 0) {
                        html += `<p style="color: var(--primary);"><i class="fas fa-check-circle"></i> Working endpoints: ${result.working_endpoints.join(', ')}</p>`;
                        html += `<p><strong>Recommended endpoint:</strong> ${result.recommended_endpoint}</p>`;
                    } else {
                        html += `<p style="color: var(--danger);"><i class="fas fa-exclamation-triangle"></i> No working endpoints found!</p>`;
                    }
                    
                    html += '<h5>Detailed Results:</h5><ul style="text-align: left;">';
                    for (const [endpoint, testResult] of Object.entries(result.test_results)) {
                        if (typeof testResult === 'object') {
                            const status = testResult.success ? '✅' : '❌';
                            html += `<li>${status} ${endpoint}: ${JSON.stringify(testResult)}</li>`;
                        }
                    }
                    html += '</ul>';
                    
                    document.getElementById('esp32Preview').innerHTML = html;
                    document.getElementById('esp32DebugInfo').innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                } catch (error) {
                    document.getElementById('esp32Preview').innerHTML = `<p style="color: var(--danger);">Error testing connection: ${error.message}</p>`;
                }
            }

            async function runFullDiagnostic() {
                document.getElementById('debugResults').innerHTML = '<div class="loading"></div> Running full diagnostic...';
                
                try {
                    const response = await fetch(API_BASE + '/test-esp32');
                    const result = await response.json();
                    
                    let html = '<h4>🔬 ESP32-CAM Full Diagnostic Report</h4>';
                    html += `<p><strong>ESP32 URL:</strong> ${result.esp32_url}</p>`;
                    
                    // Check base URL reachability
                    if (result.test_results.base_url_reachable) {
                        const base = result.test_results.base_url_reachable;
                        html += `<p><strong>Base URL Reachable:</strong> ${base.reachable ? '✅ Yes' : '❌ No'}`;
                        if (base.status_code) html += ` (Status: ${base.status_code})`;
                        if (base.error) html += ` - Error: ${base.error}`;
                        html += '</p>';
                    }
                    
                    // List working endpoints
                    if (result.working_endpoints && result.working_endpoints.length > 0) {
                        html += '<p><strong>✅ Working Endpoints:</strong></p><ul>';
                        result.working_endpoints.forEach(ep => {
                            html += `<li>${ep}</li>`;
                        });
                        html += '</ul>';
                        
                        // Add direct test button
                        html += `<button onclick="testSpecificEndpoint('${result.recommended_endpoint}')" style="margin: 10px 0;">
                            <i class="fas fa-camera"></i> Test with ${result.recommended_endpoint}
                        </button>`;
                    } else {
                        html += '<p><strong>❌ No Working Endpoints Found</strong></p>';
                        html += '<p>Troubleshooting tips:</p>';
                        html += '<ul>';
                        html += '<li>Check if ESP32-CAM is powered on</li>';
                        html += '<li>Verify IP address is correct</li>';
                        html += '<li>Check network connectivity</li>';
                        html += '<li>Ensure ESP32 is running CameraWebServer</li>';
                        html += '<li>Try accessing ' + ESP32_URL + ' directly in browser</li>';
                        html += '</ul>';
                    }
                    
                    document.getElementById('debugResults').innerHTML = html;
                } catch (error) {
                    document.getElementById('debugResults').innerHTML = `<p style="color: var(--danger);">Diagnostic error: ${error.message}</p>`;
                }
            }

            function testSpecificEndpoint(endpoint) {
                window.open(ESP32_URL + endpoint, '_blank');
            }

            function displayResults(result) {
                if (result.status !== 'success') {
                    alert('Analysis failed: ' + result.error);
                    return;
                }

                const resultsContent = document.getElementById('resultsContent');
                const confidencePercent = (result.confidence * 100).toFixed(1);
                
                resultsContent.innerHTML = `
                    <div class="disease-card" style="border-left-color: ${result.disease_info.color}">
                        <h3 style="color: ${result.disease_info.color}">${result.disease_info.display_name}</h3>
                        <p><strong><i class="fas fa-chart-bar"></i> Confidence:</strong> ${confidencePercent}%</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                        <p><strong><i class="fas fa-exclamation-triangle"></i> Severity:</strong> ${result.disease_info.severity}</p>
                        <p><strong><i class="fas fa-clock"></i> Response Time:</strong> ${result.response_time}</p>
                    </div>
                    
                    <div class="disease-card">
                        <h4><i class="fas fa-info-circle"></i> Disease Description</h4>
                        <p>${result.disease_info.description}</p>
                    </div>
                    
                    <div class="disease-card">
                        <h4><i class="fas fa-first-aid"></i> Immediate Actions</h4>
                        <ul style="text-align: left; margin-left: 20px;">
                            ${result.disease_info.immediate_actions.map(action => `<li>${action}</li>`).join('')}
                        </ul>
                    </div>
                    
                    <div class="disease-card">
                        <h4><i class="fas fa-prescription-bottle"></i> Treatment Protocol</h4>
                        <p><strong>Chemical Treatment:</strong> ${result.disease_info.treatment}</p>
                    </div>
                    
                    <div class="disease-card">
                        <h4><i class="fas fa-shield-alt"></i> Prevention Strategy</h4>
                        <ul style="text-align: left; margin-left: 20px;">
                            ${result.disease_info.prevention.map(prevention => `<li>${prevention}</li>`).join('')}
                        </ul>
                    </div>
                    
                    <div class="disease-card">
                        <h4><i class="fas fa-chart-pie"></i> All Probabilities</h4>
                        ${Object.entries(result.all_probabilities).map(([disease, prob]) => `
                            <div style="margin: 15px 0;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>${disease}</span>
                                    <span style="font-weight: bold;">${(prob * 100).toFixed(1)}%</span>
                                </div>
                                <div style="height: 8px; background: #e0e0e0; border-radius: 4px;">
                                    <div style="height: 100%; background: ${prob === result.confidence ? result.disease_info.color : '#4CAF50'}; width: ${prob * 100}%; border-radius: 4px;"></div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;

                document.getElementById('resultSection').style.display = 'block';
                document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
            }

            // Initialize on page load
            window.addEventListener('load', function() {
                initializeESP32Stream();
                // Auto-start stream
                setTimeout(() => {
                    if (document.getElementById('esp32').classList.contains('active')) {
                        startESP32Stream();
                    }
                }, 1000);
            });
        </script>
    </body>
    </html>
    ''', esp32_url=ESP32_CAM_URL)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided", "status": "error"}), 400
    
    file = request.files['file']
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        x = prepare(img)
        
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = idx2name[pred_idx]
        confidence = float(probs[pred_idx])
        
        disease_info = get_disease_info(pred_label)
        
        result = {
            "prediction": pred_label,
            "confidence": confidence,
            "all_probabilities": {idx2name[i]: float(round(p, 4)) for i, p in enumerate(probs)},
            "disease_info": disease_info,
            "response_time": get_response_time(disease_info['severity']),
            "status": "success"
        }
        
        print(f"✅ Prediction: {pred_label} ({confidence:.2%})")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/capture-from-esp32', methods=['GET'])
def capture_from_esp32_endpoint():
    """Capture image from ESP32-CAM and analyze"""
    try:
        print("🔄 Capturing image from ESP32-CAM...")
        captured_img = capture_from_esp32()
        if not captured_img:
            return jsonify({
                "error": "Failed to capture from ESP32-CAM. Please run diagnostic test to find working endpoint.", 
                "status": "error"
            }), 500
        
        print("✅ Image captured, converting to base64...")
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        captured_img.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        print("🔄 Analyzing image with ML model...")
        
        x = prepare(captured_img)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = idx2name[pred_idx]
        confidence = float(probs[pred_idx])
        
        disease_info = get_disease_info(pred_label)
        
        result = {
            "prediction": pred_label,
            "confidence": confidence,
            "all_probabilities": {idx2name[i]: float(round(p, 4)) for i, p in enumerate(probs)},
            "disease_info": disease_info,
            "response_time": get_response_time(disease_info['severity']),
            "image_data": img_base64,
            "status": "success"
        }
        
        print(f"✅ ESP32 Prediction: {pred_label} ({confidence:.2%})")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ ESP32 capture error: {e}")
        return jsonify({"error": f"ESP32-CAM processing error: {str(e)}", "status": "error"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": True,
        "esp32_url": ESP32_CAM_URL,
        "supported_diseases": list(DISEASE_INFO.keys())
    })

if __name__ == "__main__":
    print("🚀 Starting AgriSense Flask API on http://localhost:5000")
    print("🌿 ML Model: Ready")
    print(f"📡 ESP32-CAM URL: {ESP32_CAM_URL}")
    print("📸 Available capture methods:")
    print("   • Upload Image")
    print("   • Device Camera") 
    print("   • ESP32-CAM Live Stream")
    print("   • Debug Tools")
    print("\n💡 Tip: If ESP32 capture fails, go to Debug tab and run diagnostic")
    app.run(host="127.0.0.1", port=5000, debug=True)