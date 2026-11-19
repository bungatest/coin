// Rupiah Coin Scanner - MobileNetV2 Real-Time Frontend Controller

const DOM = {
  webcam: document.getElementById('webcam'),
  canvas: document.getElementById('captureCanvas'),
  aiStatusBadge: document.getElementById('aiStatusBadge'),
  statusText: document.getElementById('statusText'),
  viewfinder: document.getElementById('viewfinder'),
  viewfinderHint: document.getElementById('viewfinderHint'),
  detectionBadge: document.getElementById('detectionBadge'),
  badgeCircle: document.getElementById('badgeCircle'),
  nominalBadge: document.getElementById('nominalBadge'),
  badgeNominalVal: document.getElementById('badgeNominalVal'),
  resultNominal: document.getElementById('resultNominal'),
  confText: document.getElementById('confText'),
  speedText: document.getElementById('speedText'),
  btnSwitchCamera: document.getElementById('btnSwitchCamera'),
  btnUploadImage: document.getElementById('btnUploadImage'),
  fileInput: document.getElementById('fileInput'),
  btnVoiceToggle: document.getElementById('btnVoiceToggle'),
  btnTorchToggle: document.getElementById('btnTorchToggle'),
  iconVolumeOn: document.getElementById('iconVolumeOn'),
  iconVolumeOff: document.getElementById('iconVolumeOff'),
  probDrawer: document.getElementById('probDrawer'),
  probToggle: document.getElementById('probToggle'),
};

let currentStream = null;
let currentFacingMode = 'environment'; // default rear camera
let isScanning = false;
let scanInterval = null;
let isTorchOn = false;
let isVoiceActive = true;
let lastSpokenNominal = null;
let lastSpokenTime = 0;
let consecutiveMatches = 0;
let lastPrediction = null;

// Initialize Camera
async function startCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }

  const constraints = {
    audio: false,
    video: {
      facingMode: currentFacingMode,
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    currentStream = stream;
    DOM.webcam.srcObject = stream;
    await DOM.webcam.play();
    DOM.statusText.textContent = "AI Ready (MobileNetV2)";
    startAutoScanner();
  } catch (err) {
    console.warn("Camera init error or fallback:", err);
    DOM.statusText.textContent = "Kamera non-aktif (Upload mode)";
    DOM.viewfinderHint.textContent = "Klik icon upload di atas untuk memilih foto koin";
  }
}

// Switch Camera
DOM.btnSwitchCamera.addEventListener('click', async () => {
  currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
  await startCamera();
});

// Torch / Flashlight Toggle
DOM.btnTorchToggle.addEventListener('click', async () => {
  if (!currentStream) return;
  const track = currentStream.getVideoTracks()[0];
  const capabilities = track.getCapabilities ? track.getCapabilities() : {};

  if (!capabilities.torch) {
    alert("Lampu flash tidak didukung pada browser/kamera perangkat ini.");
    return;
  }

  try {
    isTorchOn = !isTorchOn;
    await track.applyConstraints({
      advanced: [{ torch: isTorchOn }]
    });
    DOM.btnTorchToggle.classList.toggle('active', isTorchOn);
  } catch (err) {
    console.error("Torch error:", err);
  }
});

// Voice Toggle
DOM.btnVoiceToggle.addEventListener('click', () => {
  isVoiceActive = !isVoiceActive;
  DOM.btnVoiceToggle.classList.toggle('muted', !isVoiceActive);
  DOM.iconVolumeOn.style.display = isVoiceActive ? 'block' : 'none';
  DOM.iconVolumeOff.style.display = isVoiceActive ? 'none' : 'block';
  if (!isVoiceActive && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
});

// File Upload Fallback
DOM.btnUploadImage.addEventListener('click', () => {
  DOM.fileInput.click();
});

DOM.fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  DOM.statusText.textContent = "Menganalisis foto...";
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error("Gagal memproses gambar");
    const data = await res.json();
    handlePredictionResult(data);
    DOM.statusText.textContent = "AI Ready (MobileNetV2)";
  } catch (err) {
    console.error("Upload inference error:", err);
    DOM.statusText.textContent = "Error analisis foto";
  }
});

// Capture Center Viewfinder Frame
function captureViewfinderROI() {
  if (!DOM.webcam.videoWidth || !DOM.webcam.videoHeight) return null;

  const vw = DOM.webcam.videoWidth;
  const vh = DOM.webcam.videoHeight;
  const minDim = Math.min(vw, vh);

  // Crop square center 70% matching reticle
  const cropSize = Math.floor(minDim * 0.70);
  const sx = Math.floor((vw - cropSize) / 2);
  const sy = Math.floor((vh - cropSize) / 2);

  const canvas = DOM.canvas;
  const targetSize = 256;
  canvas.width = targetSize;
  canvas.height = targetSize;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(DOM.webcam, sx, sy, cropSize, cropSize, 0, 0, targetSize, targetSize);

  return canvas.toDataURL('image/jpeg', 0.85);
}

// Auto Scanner Loop
function startAutoScanner() {
  if (isScanning) return;
  isScanning = true;

  async function loop() {
    if (!isScanning) return;

    const base64Data = captureViewfinderROI();
    if (base64Data) {
      try {
        const response = await fetch('/api/predict-base64', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: base64Data,
            crop_center_ratio: 1.0 // already cropped in canvas
          })
        });

        if (response.ok) {
          const result = await response.json();
          handlePredictionResult(result);
        }
      } catch (err) {
        console.warn("Scan frame error:", err);
      }
    }

    // Schedule next frame (~600ms latency/interval)
    scanInterval = setTimeout(loop, 600);
  }

  loop();
}

// Handle Prediction Response
function handlePredictionResult(data) {
  const { prediction, display, spoken, color, confidence, is_confident, probabilities, inference_time_ms } = data;

  // Update probabilities bars
  if (probabilities) {
    for (const [cls, prob] of Object.entries(probabilities)) {
      const pct = Math.round(prob * 100);
      const bar = document.getElementById(`bar-${cls}`);
      const val = document.getElementById(`val-${cls}`);
      if (bar) bar.style.width = `${pct}%`;
      if (val) val.textContent = `${pct}%`;
    }
  }

  DOM.speedText.textContent = `${inference_time_ms} ms`;

  // Stabilize detection with confidence threshold
  if (is_confident && confidence >= 0.65) {
    if (prediction === lastPrediction) {
      consecutiveMatches++;
    } else {
      consecutiveMatches = 1;
      lastPrediction = prediction;
    }

    // Update Result Pill
    DOM.resultNominal.textContent = display;
    DOM.badgeNominalVal.textContent = display.replace('Rp ', '');
    DOM.nominalBadge.style.backgroundColor = color || '#7C3AED';
    DOM.badgeCircle.style.backgroundColor = color || '#7C3AED';
    DOM.confText.textContent = `Akurasi ${(confidence * 100).toFixed(1)}%`;
    DOM.confText.style.color = '#059669';

    // Show checkmark badge
    DOM.detectionBadge.classList.add('active');
    DOM.viewfinderHint.textContent = `Terdeteksi: ${display}`;

    // Speak out nominal via Text-to-Speech (debounced)
    const now = Date.now();
    if (isVoiceActive && (prediction !== lastSpokenNominal || now - lastSpokenTime > 3500)) {
      speakNominal(spoken);
      lastSpokenNominal = prediction;
      lastSpokenTime = now;
    }
  } else {
    // Low confidence / No coin in reticle
    consecutiveMatches = 0;
    DOM.confText.textContent = "Scanning...";
    DOM.confText.style.color = "#94A3B8";
    DOM.detectionBadge.classList.remove('active');
    DOM.viewfinderHint.textContent = "Posisikan koin di dalam kotak";
  }
}

// Text-to-Speech Narration
function speakNominal(text) {
  if (!('speechSynthesis' in window)) return;

  window.speechSynthesis.cancel(); // cancel pending speech
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'id-ID';
  utterance.rate = 1.05;
  utterance.pitch = 1.0;

  // Try to pick an Indonesian voice if available
  const voices = window.speechSynthesis.getVoices();
  const idVoice = voices.find(v => v.lang.startsWith('id') || v.lang.includes('ID'));
  if (idVoice) {
    utterance.voice = idVoice;
  }

  window.speechSynthesis.speak(utterance);
}

// Toggle Probability Drawer
DOM.probToggle.addEventListener('click', () => {
  DOM.probDrawer.classList.toggle('collapsed');
});

// Window startup
window.addEventListener('DOMContentLoaded', () => {
  startCamera();
  if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => {};
  }
});
