// Rupiah Coin Scanner - In-Browser TensorFlow.js WebGL (Direct Client AI)

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

// 1. Inisialisasi Model AI TensorFlow.js
async function initAIModel() {
  DOM.statusText.textContent = "Memuat AI WebGL...";
  
  // Tunggu pustaka TFJS jika belum siap
  let waitCount = 0;
  while (typeof tf === 'undefined' && waitCount < 25) {
    await new Promise(r => setTimeout(r, 200));
    waitCount++;
  }

  if (typeof tf === 'undefined') {
    DOM.statusText.textContent = "Error: TFJS CDN gagal dimuat";
    console.error("TensorFlow.js library not loaded");
    return;
  }

  try {
    await tf.ready();
    try {
      await tf.setBackend('webgl');
    } catch (e) {
      console.warn("WebGL unavailable, using CPU:", e);
      await tf.setBackend('cpu');
    }

    const modelLocations = [
      "model_coin_tfjs/model.json",
      "./model_coin_tfjs/model.json",
      "/coin/model_coin_tfjs/model.json",
      "deploy/model_coin_tfjs/model.json"
    ];

    let lastError = null;
    for (const loc of modelLocations) {
      try {
        console.log("[AI Engine] Memuat model dari:", loc);
        tfModel = await tf.loadLayersModel(loc, { requestInit: { cache: "no-store" } });
        if (tfModel) {
          console.log("[AI Engine] Sukses memuat model dari:", loc);
          break;
        }
      } catch (err) {
        lastError = err;
        console.warn(`[AI Engine] Gagal dari ${loc}:`, err);
      }
    }

    if (tfModel) {
      // Warmup forward pass untuk kompilasi shader WebGL
      tf.tidy(() => {
        const dummy = tf.zeros([1, 224, 224, 3]);
        tfModel.predict(dummy);
      });
      modelReady = true;
      const backendName = tf.getBackend().toUpperCase();
      DOM.statusText.textContent = `AI Ready (${backendName})`;
      console.log(`[AI Engine] MobileNetV2 Koin Siap Aktif (${backendName})!`);
      return;
    } else {
      throw lastError || new Error("File model_coin_tfjs/model.json tidak ditemukan");
    }
  } catch (err) {
    console.error("[AI Engine] Error inisialisasi model:", err);
    DOM.statusText.textContent = "Error memuat model AI";
  }
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
    DOM.viewfinderHint.textContent = "Klik tombol upload di atas untuk memilih foto koin";
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

// Upload Foto Koin
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

  // Crop bujur sangkar di tengah reticle (70% dari minDim)
  const cropSize = Math.floor(minDim * 0.70);
  const sx = Math.floor((vw - cropSize) / 2);
  const sy = Math.floor((vh - cropSize) / 2);

  const canvas = DOM.canvas;
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(DOM.webcam, sx, sy, cropSize, cropSize, 0, 0, 224, 224);

  return canvas;
}

// Auto Scanner Loop (Realtime 350ms)
function startAutoScanner() {
  if (isScanning) return;
  isScanning = true;

  async function loop() {
    if (!isScanning) return;

    if (modelReady && tfModel) {
      const canvas = prepareViewfinderCanvas();
      if (canvas) {
        await predictFromCanvas(canvas);
      }
    }

    setTimeout(loop, 350);
  }

  loop();
}

// Inferensi langsung di browser menggunakan TensorFlow.js WebGL
async function predictFromCanvas(canvas) {
  const t0 = performance.now();
  let tensor = null;
  let prediction = null;

  try {
    // Normalisasi MobileNetV2: [0, 255] -> [-1, 1]
    tensor = tf.tidy(() => {
      return tf.browser.fromPixels(canvas)
        .toFloat()
        .div(127.5)
        .sub(1.0)
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

    // Filter deteksi koin (threshold 0.50)
    if (maxProb >= 0.50) {
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
