// Rupiah Coin Scanner - Dual Engine (TensorFlow.js WebGL In-Browser + Flask API Fallback)

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

const COIN_CLASSES = {
  0: { code: 'Rp100', nominal: 100, display: 'Rp 100', spoken: 'Seratus Rupiah', color: '#3B82F6', barId: 'bar-Rp100', valId: 'val-Rp100' },
  1: { code: 'Rp1000', nominal: 1000, display: 'Rp 1.000', spoken: 'Seribu Rupiah', color: '#F59E0B', barId: 'bar-Rp1000', valId: 'val-Rp1000' },
  2: { code: 'Rp200', nominal: 200, display: 'Rp 200', spoken: 'Dua Ratus Rupiah', color: '#10B981', barId: 'bar-Rp200', valId: 'val-Rp200' },
  3: { code: 'Rp500', nominal: 500, display: 'Rp 500', spoken: 'Lima Ratus Rupiah', color: '#8B5CF6', barId: 'bar-Rp500', valId: 'val-Rp500' },
  4: { code: 'garuda', nominal: 0, display: 'Sisi Garuda', spoken: 'Sisi Garuda, balik koinnya', color: '#EC4899', barId: 'bar-garuda', valId: 'val-garuda' },
};

let tfModel = null;
let modelReady = false;
let currentStream = null;
let currentFacingMode = 'environment';
let isScanning = false;
let isTorchOn = false;
let isVoiceActive = true;
let lastSpokenNominal = null;
let lastSpokenTime = 0;
let lastPrediction = null;

// 1. Inisialisasi Model AI TensorFlow.js (Berjalan 100% di Browser / GitHub Pages)
async function initAIModel() {
  try {
    DOM.statusText.textContent = "Memuat AI WebGL...";
    if (typeof tf !== 'undefined') {
      await tf.ready();
      try {
        await tf.setBackend('webgl');
      } catch (e) {
        await tf.setBackend('cpu');
      }
      
      const modelLocations = [
        "model_coin_tfjs/model.json",
        "./model_coin_tfjs/model.json",
        "deploy/model_coin_tfjs/model.json",
        "/coin/model_coin_tfjs/model.json"
      ];

      for (const loc of modelLocations) {
        try {
          console.log("[AI Engine] Mencoba memuat model dari:", loc);
          tfModel = await tf.loadLayersModel(loc, { requestInit: { cache: "no-store" } });
          if (tfModel) break;
        } catch (err) {
          // coba lokasi berikutnya
        }
      }

      if (tfModel) {
        // Warmup model forward pass
        tf.tidy(() => {
          const dummy = tf.zeros([1, 224, 224, 3]);
          tfModel.predict(dummy);
        });
        modelReady = true;
        const backendName = tf.getBackend().toUpperCase();
        DOM.statusText.textContent = `AI Ready (${backendName})`;
        console.log(`[AI Engine] Model loaded successfully with ${backendName} backend!`);
        return;
      }
    }
  } catch (err) {
    console.warn("[AI Engine] TFJS model load error, fallback ke API lokal:", err);
  }

  // Jika TFJS tidak dapat dimuat, gunakan fallback API backend
  DOM.statusText.textContent = "AI Ready (API Mode)";
}

// 2. Inisialisasi Kamera
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
    startAutoScanner();
  } catch (err) {
    console.warn("Camera init error or fallback:", err);
    DOM.statusText.textContent = "Kamera non-aktif (Upload mode)";
    DOM.viewfinderHint.textContent = "Klik tombol upload di atas untuk menguji foto koin";
  }
}

// Ganti Kamera (Depan / Belakang)
DOM.btnSwitchCamera.addEventListener('click', async () => {
  currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
  await startCamera();
});

// Senter / Flashlight
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

// Suara Toggle
DOM.btnVoiceToggle.addEventListener('click', () => {
  isVoiceActive = !isVoiceActive;
  DOM.btnVoiceToggle.classList.toggle('muted', !isVoiceActive);
  DOM.iconVolumeOn.style.display = isVoiceActive ? 'block' : 'none';
  DOM.iconVolumeOff.style.display = isVoiceActive ? 'none' : 'block';
  if (!isVoiceActive && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
});

// Upload Foto Koin (Fallback)
DOM.btnUploadImage.addEventListener('click', () => {
  DOM.fileInput.click();
});

DOM.fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = async () => {
    const canvas = DOM.canvas;
    const ctx = canvas.getContext('2d');
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(img, 0, 0, 224, 224);

    if (modelReady && tfModel) {
      await predictFromCanvas(canvas);
    } else {
      // Fallback via API backend
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();
        handlePredictionResult(data);
      } catch (err) {
        console.error("API error:", err);
      }
    }
  };
  img.src = URL.createObjectURL(file);
});

// Crop area viewfinder ke 224x224 canvas
function prepareViewfinderCanvas() {
  if (!DOM.webcam.videoWidth || !DOM.webcam.videoHeight) return null;

  const vw = DOM.webcam.videoWidth;
  const vh = DOM.webcam.videoHeight;
  const minDim = Math.min(vw, vh);

  // Crop bujur sangkar di tengah reticle (72% dari minDim)
  const cropSize = Math.floor(minDim * 0.72);
  const sx = Math.floor((vw - cropSize) / 2);
  const sy = Math.floor((vh - cropSize) / 2);

  const canvas = DOM.canvas;
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(DOM.webcam, sx, sy, cropSize, cropSize, 0, 0, 224, 224);

  return canvas;
}

// Auto Scanner Loop (Realtime 350-500ms)
function startAutoScanner() {
  if (isScanning) return;
  isScanning = true;

  async function loop() {
    if (!isScanning) return;

    const canvas = prepareViewfinderCanvas();
    if (canvas) {
      if (modelReady && tfModel) {
        await predictFromCanvas(canvas);
      } else {
        // Fallback jika model tfjs belum siap
        await predictFromBackendAPI(canvas);
      }
    }

    setTimeout(loop, 400);
  }

  loop();
}

// Inferensi langsung di browser menggunakan TensorFlow.js
async function predictFromCanvas(canvas) {
  const t0 = performance.now();
  let tensor = null;
  let prediction = null;

  try {
    tensor = tf.tidy(() => {
      return tf.browser.fromPixels(canvas)
        .toFloat()
        .expandDims(0);
    });

    prediction = tfModel.predict(tensor);
    const probs = await prediction.data();
    const latency = Math.round(performance.now() - t0);

    // Cari probabilitas tertinggi
    let maxIdx = 0;
    let maxProb = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIdx = i;
      }
    }

    const detected = COIN_CLASSES[maxIdx] || COIN_CLASSES[0];
    
    // Update rincian probabilitas di drawer
    for (let i = 0; i < probs.length; i++) {
      const clsInfo = COIN_CLASSES[i];
      if (clsInfo) {
        const pct = Math.round(probs[i] * 100);
        const bar = document.getElementById(clsInfo.barId);
        const val = document.getElementById(clsInfo.valId);
        if (bar) bar.style.width = `${pct}%`;
        if (val) val.textContent = `${pct}%`;
      }
    }

    DOM.speedText.textContent = `${latency} ms`;

    // Filter deteksi koin (threshold 0.55)
    if (maxProb >= 0.55) {
      DOM.resultNominal.textContent = detected.display;
      DOM.badgeNominalVal.textContent = detected.nominal > 0 ? `${detected.nominal}` : 'GRD';
      DOM.nominalBadge.style.backgroundColor = detected.color;
      DOM.badgeCircle.style.backgroundColor = detected.color;
      DOM.confText.textContent = `Akurasi ${(maxProb * 100).toFixed(1)}%`;
      DOM.confText.style.color = '#059669';

      DOM.detectionBadge.classList.add('active');
      DOM.viewfinderHint.textContent = `Terdeteksi: ${detected.display}`;

      // Narasi Suara TTS
      const now = Date.now();
      if (isVoiceActive && (detected.code !== lastSpokenNominal || now - lastSpokenTime > 3500)) {
        speakNominal(detected.spoken);
        lastSpokenNominal = detected.code;
        lastSpokenTime = now;
      }
    } else {
      DOM.confText.textContent = "Scanning...";
      DOM.confText.style.color = "#94A3B8";
      DOM.detectionBadge.classList.remove('active');
      DOM.viewfinderHint.textContent = "Arahkan koin ke dalam kotak";
    }

  } catch (err) {
    console.warn("Client inference error:", err);
  } finally {
    if (tensor) tensor.dispose();
    if (prediction) prediction.dispose();
  }
}

// Fallback jika menggunakan server API backend (Localhost Flask/FastAPI)
async function predictFromBackendAPI(canvas) {
  const base64Data = canvas.toDataURL('image/jpeg', 0.85);
  try {
    const res = await fetch('/api/predict-base64', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Data, crop_center_ratio: 1.0 })
    });
    if (res.ok) {
      const data = await res.json();
      handlePredictionResult(data);
    }
  } catch (e) {
    // Abaikan jika server backend tidak online di GitHub Pages
  }
}

function handlePredictionResult(data) {
  const { display, spoken, color, confidence, is_confident, probabilities, inference_time_ms } = data;
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
  if (is_confident && confidence >= 0.55) {
    DOM.resultNominal.textContent = display;
    DOM.badgeNominalVal.textContent = display.replace('Rp ', '');
    DOM.nominalBadge.style.backgroundColor = color || '#7C3AED';
    DOM.badgeCircle.style.backgroundColor = color || '#7C3AED';
    DOM.confText.textContent = `Akurasi ${(confidence * 100).toFixed(1)}%`;
    DOM.confText.style.color = '#059669';
    DOM.detectionBadge.classList.add('active');
    DOM.viewfinderHint.textContent = `Terdeteksi: ${display}`;
    const now = Date.now();
    if (isVoiceActive && (display !== lastSpokenNominal || now - lastSpokenTime > 3500)) {
      speakNominal(spoken);
      lastSpokenNominal = display;
      lastSpokenTime = now;
    }
  }
}

// Text-to-Speech Narration
function speakNominal(text) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'id-ID';
  utterance.rate = 1.05;
  utterance.pitch = 1.0;
  const voices = window.speechSynthesis.getVoices();
  const idVoice = voices.find(v => v.lang.startsWith('id') || v.lang.includes('ID'));
  if (idVoice) utterance.voice = idVoice;
  window.speechSynthesis.speak(utterance);
}

// Toggle Probability Drawer
DOM.probToggle.addEventListener('click', () => {
  DOM.probDrawer.classList.toggle('collapsed');
});

// Startup
window.addEventListener('DOMContentLoaded', async () => {
  await initAIModel();
  await startCamera();
});
