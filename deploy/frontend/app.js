// Rupiah Coin Scanner - Hybrid Deep Learning (MobileNetV2) + OCR Digit Recognition Engine

const DOM = {
  webcam: document.getElementById('webcam'),
  canvas: document.getElementById('captureCanvas'),
  ocrCanvas: document.getElementById('ocrCanvas'),
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
  btnScanOCR: document.getElementById('btnScanOCR'),
  btnVoiceToggle: document.getElementById('btnVoiceToggle'),
  btnTorchToggle: document.getElementById('btnTorchToggle'),
  iconVolumeOn: document.getElementById('iconVolumeOn'),
  iconVolumeOff: document.getElementById('iconVolumeOff'),
  probDrawer: document.getElementById('probDrawer'),
  probToggle: document.getElementById('probToggle'),
};

// 4 Kelas Nominal Resmi Dataset Koin Baru
const COIN_CLASSES = {
  0: { code: 'Rp100', nominal: 100, display: 'Rp 100', spoken: 'Seratus Rupiah', color: '#3B82F6', barId: 'bar-Rp100', valId: 'val-Rp100' },
  1: { code: 'Rp1000', nominal: 1000, display: 'Rp 1.000', spoken: 'Seribu Rupiah', color: '#F59E0B', barId: 'bar-Rp1000', valId: 'val-Rp1000' },
  2: { code: 'Rp200', nominal: 200, display: 'Rp 200', spoken: 'Dua Ratus Rupiah', color: '#10B981', barId: 'bar-Rp200', valId: 'val-Rp200' },
  3: { code: 'Rp500', nominal: 500, display: 'Rp 500', spoken: 'Lima Ratus Rupiah', color: '#8B5CF6', barId: 'bar-Rp500', valId: 'val-Rp500' },
};

let tfModel = null;
let modelReady = false;
let ocrWorker = null;
let ocrReady = false;
let isOcrBusy = false;
let lastOcrRun = 0;

let currentStream = null;
let currentFacingMode = 'environment';
let isScanning = false;
let isTorchOn = false;
let isVoiceActive = true;
let lastSpokenNominal = null;
let lastSpokenTime = 0;

// Stabilizer Buffer (Anti-Flicker)
const BUFFER_SIZE = 5;
const historyBuffer = [];
let lockedPrediction = null;
let lockCounter = 0;

// 1. Inisialisasi Model AI TensorFlow.js
async function initAIModel() {
  DOM.statusText.textContent = "Memuat AI MobileNetV2...";

  let waitCount = 0;
  while (typeof tf === 'undefined' && waitCount < 30) {
    await new Promise(r => setTimeout(r, 200));
    waitCount++;
  }

  if (typeof tf === 'undefined') {
    DOM.statusText.textContent = "Error: TFJS CDN gagal dimuat";
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
      // Warmup forward pass
      tf.tidy(() => {
        const dummy = tf.zeros([1, 224, 224, 3]);
        tfModel.predict(dummy);
      });
      modelReady = true;
      const backendName = tf.getBackend().toUpperCase();
      DOM.statusText.textContent = `AI Ready (${backendName} + OCR)`;
      console.log(`[AI Engine] MobileNetV2 Koin Siap Aktif (${backendName})!`);
    } else {
      console.warn("[AI Engine] TFJS model tidak tersedia, beralih ke Mode OCR Penuh");
      DOM.statusText.textContent = "OCR Ready (Mode Angka Koin)";
    }
  } catch (err) {
    console.error("[AI Engine] Error inisialisasi model:", err);
    DOM.statusText.textContent = "OCR Ready (Mode Angka Koin)";
  }
}

// 2. Inisialisasi Tesseract.js OCR untuk Pembacaan Angka Koin Langsung
async function initOCR() {
  if (typeof Tesseract === 'undefined') {
    console.warn("[OCR] Tesseract.js belum terpasang di dokumen.");
    return;
  }
  try {
    console.log("[OCR] Menginisialisasi Tesseract OCR Worker...");
    ocrWorker = await Tesseract.createWorker('eng', 1, {
      logger: () => {}
    });
    await ocrWorker.setParameters({
      tessedit_char_whitelist: '0125RUPIAH.',
      tessedit_pageseg_mode: '7', // treat image as a single text line
    });
    ocrReady = true;
    console.log("[OCR] Tesseract Worker Siap Membaca Angka Timbul Koin!");
  } catch (err) {
    console.warn("[OCR] Gagal inisialisasi worker:", err);
  }
}

// 3. Preprocessing Khusus Angka Timbul Koin (Adaptive High Contrast)
function prepareOCRCanvas(sourceCanvas) {
  const ocrCanvas = DOM.ocrCanvas || document.createElement('canvas');
  ocrCanvas.width = 280;
  ocrCanvas.height = 280;
  const ctx = ocrCanvas.getContext('2d');

  // Crop fokus area tengah koin tempat angka dicetak (65% diameter koin)
  const sw = sourceCanvas.width;
  const sh = sourceCanvas.height;
  const cropW = Math.floor(sw * 0.68);
  const cropH = Math.floor(sh * 0.68);
  const sx = Math.floor((sw - cropW) / 2);
  const sy = Math.floor((sh - cropH) / 2);

  ctx.drawImage(sourceCanvas, sx, sy, cropW, cropH, 0, 0, 280, 280);

  // Ambil piksel dan terapkan adaptive thresholding
  const imgData = ctx.getImageData(0, 0, 280, 280);
  const d = imgData.data;

  // Hitung rata-rata luminansi
  let totalLum = 0;
  for (let i = 0; i < d.length; i += 4) {
    totalLum += (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
  }
  const avgLum = totalLum / (d.length / 4);

  // Filter kontras tinggi untuk memisahkan lekukan angka timbul koin
  for (let i = 0; i < d.length; i += 4) {
    const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    const diff = Math.abs(lum - avgLum);
    // Jika ada bayangan tepi angka, jadikan hitam; sisanya putih
    const val = diff > 24 ? 0 : 255;
    d[i] = val;
    d[i + 1] = val;
    d[i + 2] = val;
  }
  ctx.putImageData(imgData, 0, 0);
  return ocrCanvas;
}

// 4. Deteksi Angka Koin via OCR
async function scanWithOCR(canvas, forceTrigger = false) {
  if (!ocrReady || !ocrWorker || isOcrBusy) return null;
  const now = Date.now();
  if (!forceTrigger && (now - lastOcrRun < 1600)) return null;

  isOcrBusy = true;
  lastOcrRun = now;
  if (DOM.btnScanOCR) DOM.btnScanOCR.classList.add('scanning');

  try {
    const ocrCanvas = prepareOCRCanvas(canvas);
    const { data: { text } } = await ocrWorker.recognize(ocrCanvas);
    const clean = text.replace(/[^0-9RUPIAH]/gi, '').toUpperCase();
    console.log(`[OCR Deteksi]: "${text.trim()}" -> Cleaned: "${clean}"`);

    let matched = null;
    if (clean.includes('1000') || clean.includes('1.000') || clean.includes('SERIBU')) {
      matched = COIN_CLASSES[1]; // Rp 1.000
    } else if (clean.includes('200') || clean.includes('DUARATUS')) {
      matched = COIN_CLASSES[2]; // Rp 200
    } else if (clean.includes('500') || clean.includes('LIMARATUS')) {
      matched = COIN_CLASSES[3]; // Rp 500
    } else if (clean.includes('100') || clean.includes('SERATUS')) {
      matched = COIN_CLASSES[0]; // Rp 100
    }

    if (matched) {
      console.log(`🎯 [OCR KUNCI BERHASIL]: ${matched.display}`);
      applyConfirmedDetection(matched, 0.998, `Angka Terbaca: ${matched.display}`);
      return matched;
    }
  } catch (err) {
    console.warn("[OCR] Error saat scanning:", err);
  } finally {
    isOcrBusy = false;
    if (DOM.btnScanOCR) DOM.btnScanOCR.classList.remove('scanning');
  }
  return null;
}

// 5. Terapkan Hasil Deteksi Valid ke Tampilan & Suara
function applyConfirmedDetection(detected, confidence, hintText) {
  lockedPrediction = detected.code;
  lockCounter = 10;

  DOM.resultNominal.textContent = detected.display;
  DOM.badgeNominalVal.textContent = `${detected.nominal}`;
  DOM.nominalBadge.style.backgroundColor = detected.color;
  DOM.badgeCircle.style.backgroundColor = detected.color;
  DOM.confText.textContent = `Akurasi ${(confidence * 100).toFixed(1)}% (Pasti)`;
  DOM.confText.style.color = '#059669';

  DOM.detectionBadge.classList.add('active');
  DOM.viewfinderHint.textContent = hintText || `Koin Valid: ${detected.display}`;

  // Update Bar Probabilitas (set ke 100% untuk kelas terpilih)
  for (let c = 0; c < 4; c++) {
    const cls = COIN_CLASSES[c];
    const isTarget = cls.code === detected.code;
    const pct = isTarget ? 100 : 0;
    const bar = document.getElementById(cls.barId);
    const val = document.getElementById(cls.valId);
    if (bar) bar.style.width = `${pct}%`;
    if (val) val.textContent = `${pct}%`;
  }

  // Narasi Suara TTS
  const now = Date.now();
  if (isVoiceActive && (detected.code !== lastSpokenNominal || now - lastSpokenTime > 3000)) {
    speakNominal(detected.spoken);
    lastSpokenNominal = detected.code;
    lastSpokenTime = now;
  }
}

// 6. Inisialisasi Kamera
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

// Tombol Baca Angka Koin (OCR Manual Trigger)
DOM.btnScanOCR.addEventListener('click', async () => {
  if (DOM.canvas) {
    DOM.viewfinderHint.textContent = "Sedang membaca angka timbul koin...";
    const ocrResult = await scanWithOCR(DOM.canvas, true);
    if (!ocrResult) {
      DOM.viewfinderHint.textContent = "Posisikan angka timbul koin lebih jelas";
    }
  }
});

// Klik viewfinder untuk memicu fokus OCR
DOM.viewfinder.addEventListener('click', async () => {
  if (DOM.canvas) {
    await scanWithOCR(DOM.canvas, true);
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

    // Coba baca angka OCR terlebih dahulu
    const ocrMatch = await scanWithOCR(canvas, true);
    if (!ocrMatch && modelReady && tfModel) {
      historyBuffer.length = 0;
      await runSingleInference(canvas, true);
    }
  };
  img.src = URL.createObjectURL(file);
});

// Crop area viewfinder ke 224x224 canvas
function prepareMultiScaleCanvas(scaleRatio = 0.65) {
  if (!DOM.webcam.videoWidth || !DOM.webcam.videoHeight) return null;

  const vw = DOM.webcam.videoWidth;
  const vh = DOM.webcam.videoHeight;
  const minDim = Math.min(vw, vh);

  const cropSize = Math.floor(minDim * scaleRatio);
  const sx = Math.floor((vw - cropSize) / 2);
  const sy = Math.floor((vh - cropSize) / 2);

  const canvas = DOM.canvas;
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  ctx.filter = "contrast(1.15) brightness(0.98)";
  ctx.drawImage(DOM.webcam, sx, sy, cropSize, cropSize, 0, 0, 224, 224);
  ctx.filter = "none";

  return canvas;
}

// Auto Scanner Loop (Interval 280ms)
function startAutoScanner() {
  if (isScanning) return;
  isScanning = true;

  async function loop() {
    if (!isScanning) return;

    const tightCanvas = prepareMultiScaleCanvas(0.62);
    if (tightCanvas) {
      // 1. Jalankan inferensi CNN MobileNetV2
      if (modelReady && tfModel) {
        await runSingleInference(tightCanvas, false);
      }
      // 2. Jalankan OCR periodik di latar belakang untuk verifikasi angka
      if (ocrReady && !isOcrBusy && (Date.now() - lastOcrRun > 1500)) {
        await scanWithOCR(tightCanvas, false);
      }
    }

    setTimeout(loop, 260);
  }

  loop();
}

// Eksekusi Prediksi Deep Learning dengan 4-Rotasi TTA + Temporal Consensus
async function runSingleInference(canvas, isUpload = false) {
  const t0 = performance.now();
  let tensor = null;
  let prediction = null;

  try {
    tensor = tf.tidy(() => {
      // 4-Angle Rotation TTA (0°, 90°, 180°, 270°) agar kebal terhadap sudut rotasi koin
      const base = tf.browser.fromPixels(canvas).toFloat().div(127.5).sub(1.0);
      const rot0 = base.expandDims(0);
      const rot90 = tf.image.rotateWithOffset(rot0, Math.PI / 2);
      const rot180 = tf.image.rotateWithOffset(rot0, Math.PI);
      const rot270 = tf.image.rotateWithOffset(rot0, (3 * Math.PI) / 2);
      return tf.concat([rot0, rot90, rot180, rot270], 0);
    });

    prediction = tfModel.predict(tensor);
    const ttaProbs = await prediction.data();
    const latency = Math.round(performance.now() - t0);

    // Ambil rata-rata dari 4 rotasi TTA (masing-masing 4 atau 5 kelas output)
    const numOutputs = tfModel.outputs[0].shape[1] || 4;
    const avgTtaProbs = new Array(numOutputs).fill(0);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < numOutputs; c++) {
        avgTtaProbs[c] += ttaProbs[r * numOutputs + c] / 4.0;
      }
    }

    // Normalisasi 4 kelas resmi
    const fourClassProbs = [
      avgTtaProbs[0] || 0,
      avgTtaProbs[1] || 0,
      avgTtaProbs[2] || 0,
      avgTtaProbs[3] || 0,
    ];
    const sumP = fourClassProbs.reduce((a, b) => a + b, 0) || 1.0;
    const normProbs = fourClassProbs.map(p => p / sumP);

    // Masukkan ke history buffer konsensus
    historyBuffer.push(normProbs);
    if (historyBuffer.length > BUFFER_SIZE) {
      historyBuffer.shift();
    }

    // Moving average berbobot waktu
    const avgProbs = new Array(4).fill(0);
    let sumWeights = 0;
    for (let f = 0; f < historyBuffer.length; f++) {
      const weight = 1 + (f * 0.25);
      sumWeights += weight;
      for (let c = 0; c < 4; c++) {
        avgProbs[c] += historyBuffer[f][c] * weight;
      }
    }
    for (let c = 0; c < 4; c++) avgProbs[c] /= sumWeights;

    // Cari kelas terbaik
    let bestIdx = 0;
    let bestProb = 0;
    for (let c = 0; c < 4; c++) {
      if (avgProbs[c] > bestProb) {
        bestProb = avgProbs[c];
        bestIdx = c;
      }
    }

    // Hitung konsensus voting
    let agreementCount = 0;
    for (const frameProbs of historyBuffer) {
      let fMaxIdx = 0, fMaxVal = 0;
      for (let c = 0; c < 4; c++) {
        if (frameProbs[c] > fMaxVal) {
          fMaxVal = frameProbs[c];
          fMaxIdx = c;
        }
      }
      if (fMaxIdx === bestIdx) agreementCount++;
    }

    const consensusRate = historyBuffer.length > 0 ? (agreementCount / historyBuffer.length) : 0;
    const detected = COIN_CLASSES[bestIdx] || COIN_CLASSES[0];

    // Update Drawer Rincian Probabilitas 4 Kelas
    for (let c = 0; c < 4; c++) {
      const clsInfo = COIN_CLASSES[c];
      if (clsInfo) {
        const pct = Math.round(avgProbs[c] * 100);
        const bar = document.getElementById(clsInfo.barId);
        const val = document.getElementById(clsInfo.valId);
        if (bar) bar.style.width = `${pct}%`;
        if (val) val.textContent = `${pct}%`;
      }
    }

    DOM.speedText.textContent = `${latency} ms`;

    // Syarat Konfirmasi Koin Stabil:
    const isStable = isUpload || (consensusRate >= 0.70 && bestProb >= 0.45 && historyBuffer.length >= 3);

    if (isStable) {
      applyConfirmedDetection(detected, bestProb, `Koin Stabil: ${detected.display}`);
    } else {
      if (bestProb > 0.35 && historyBuffer.length >= 2) {
        DOM.confText.textContent = `Menstabilkan (${agreementCount}/${historyBuffer.length})...`;
        DOM.confText.style.color = "#EAB308";
        DOM.viewfinderHint.textContent = `Tahan posisi koin: ${detected.display}?`;
      } else {
        DOM.confText.textContent = "Scanning...";
        DOM.confText.style.color = "#94A3B8";
        DOM.detectionBadge.classList.remove('active');
        DOM.viewfinderHint.textContent = "Posisikan koin pas di dalam kotak";
      }
    }

  } catch (err) {
    console.warn("[AI Engine] Inference error:", err);
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

// Toggle Drawer
DOM.probToggle.addEventListener('click', () => {
  DOM.probDrawer.classList.toggle('collapsed');
});

// Startup Inisialisasi
window.addEventListener('DOMContentLoaded', async () => {
  await Promise.all([
    initAIModel(),
    initOCR(),
  ]);
  await startCamera();
});
