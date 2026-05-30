/* ── DOM ─────────────────────────────────────────────────────────────────── */
const orbEl      = document.getElementById('orb');
const canvas     = document.getElementById('orbCanvas');
const statusEl   = document.getElementById('status');
const transcript = document.getElementById('transcript');
const startBtn   = document.getElementById('startBtn');
const stopBtn    = document.getElementById('stopBtn');
const downloadBtn= document.getElementById('downloadBtn');

/* ── State ───────────────────────────────────────────────────────────────── */
let ws            = null;
let audioCtx      = null;
let nextPlayTime  = 0;
let orbState      = 'idle';   // idle | listening | speaking
let currentBotEl  = null;
let currentYouEl  = null;

/* ── Conversation log for download ──────────────────────────────────────── */
const chatLog = [];

/* ── Start ───────────────────────────────────────────────────────────────── */
startBtn.addEventListener('click', async () => {
  startBtn.classList.add('hidden');
  setStatus('Requesting microphone...');
  try {
    await initAudio();
    connectWS();
  } catch (e) {
    console.error(e);
    setStatus(`Mic error: ${e.name} — ${e.message}`);
    startBtn.classList.remove('hidden');
  }
});

/* ── Stop ────────────────────────────────────────────────────────────────── */
stopBtn.addEventListener('click', () => {
  if (ws) { ws.close(); ws = null; }
  setState('idle');
  setStatus('Conversation ended.');
  stopBtn.classList.add('hidden');
  startBtn.classList.remove('hidden');
});

/* ── Download transcript ─────────────────────────────────────────────────── */
downloadBtn.addEventListener('click', () => {
  if (!chatLog.length) return;
  const date = new Date().toLocaleString();
  const lines = [`Ayush Gautam — Interview Transcript\n${date}\n${'─'.repeat(40)}\n`];
  chatLog.forEach(m => lines.push(`${m.role === 'you' ? 'Interviewer' : 'Ayush'}:\n${m.text}\n`));
  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `ayush-interview-${Date.now()}.txt`;
  a.click();
});

/* ── WebSocket ───────────────────────────────────────────────────────────── */
function connectWS() {
  // Works locally (ws://localhost:8000/ws) and on Vercel (wss://your-app.vercel.app/ws → Railway)
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen    = () => setStatus('Connecting to Ayush...');
  ws.onclose   = () => {
    setState('idle');
    setStatus('Conversation ended.');
    stopBtn.classList.add('hidden');
    startBtn.classList.remove('hidden');
  };
  ws.onerror   = () => setStatus('Connection error.');
  ws.onmessage = onMessage;
}

function onMessage(ev) {
  const data = JSON.parse(ev.data);

  switch (data.type) {

    case 'ready':
      setState('idle');
      setStatus('Ready — just speak');
      stopBtn.classList.remove('hidden');
      downloadBtn.classList.add('hidden');
      // Clear empty hint
      const hint = transcript.querySelector('.empty-hint');
      if (hint) hint.remove();
      break;

    case 'speech_started':
      nextPlayTime = 0;
      setState('listening');
      setStatus('Listening...');
      currentYouEl = addMessage('you', '');
      currentBotEl = null;
      break;

    case 'user_transcript':
      if (currentYouEl) {
        currentYouEl.textContent = data.text;
        chatLog.push({ role: 'you', text: data.text });
      }
      scrollDown();
      break;

    case 'bot_transcript':
      if (!currentBotEl) {
        setState('speaking');
        setStatus('Ayush is speaking...');
        currentBotEl = addMessage('bot', '');
        chatLog.push({ role: 'bot', text: '' });
      }
      currentBotEl.textContent += data.delta;
      chatLog[chatLog.length - 1].text += data.delta;
      downloadBtn.classList.remove('hidden');
      scrollDown();
      break;

    case 'audio':
      playChunk(data.delta);
      break;

    case 'response_done':
      setState('idle');
      setStatus('Ready — just speak');
      break;

    case 'error':
      setStatus(`Error: ${data.message}`);
      setState('idle');
      break;
  }
}

/* ── Audio — mic capture ─────────────────────────────────────────────────── */
async function initAudio() {
  audioCtx = new AudioContext({ sampleRate: 24000 });

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  const source = audioCtx.createMediaStreamSource(stream);

  // ScriptProcessorNode: captures PCM32 → convert → send as PCM16 binary
  const processor = audioCtx.createScriptProcessor(2048, 1, 1);
  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const f32  = e.inputBuffer.getChannelData(0);
    const i16  = f32ToI16(f32);
    ws.send(i16.buffer);
  };

  // Silent gain so onaudioprocess fires but mic doesn't loop back to speakers
  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(audioCtx.destination);
}

function f32ToI16(f32) {
  const i16 = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return i16;
}

/* ── Audio — PCM16 playback ──────────────────────────────────────────────── */
function playChunk(b64) {
  if (!audioCtx) return;

  const binary = atob(b64);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const i16  = new Int16Array(bytes.buffer);
  const f32  = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768.0;

  const buf  = audioCtx.createBuffer(1, f32.length, 24000);
  buf.copyToChannel(f32, 0);

  const src  = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);

  const now  = audioCtx.currentTime;
  const at   = Math.max(nextPlayTime, now + 0.04);
  src.start(at);
  nextPlayTime = at + buf.duration;
}

/* ── UI helpers ──────────────────────────────────────────────────────────── */
function setState(s) {
  orbState    = s;
  orbEl.className = `orb ${s}`;
}

function setStatus(txt) { statusEl.textContent = txt; }

function addMessage(role, text) {
  const row    = document.createElement('div');
  row.className = `msg ${role}`;

  const label  = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'you' ? 'You' : 'Ayush';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;

  row.appendChild(label);
  row.appendChild(bubble);
  transcript.appendChild(row);
  scrollDown();
  return bubble;
}

function scrollDown() {
  transcript.scrollTop = transcript.scrollHeight;
}

/* ── Canvas orb animation ────────────────────────────────────────────────── */
let frame = 0;
const ctx = canvas.getContext('2d');

function resizeCanvas() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

function drawOrb() {
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const r  = Math.min(W, H) / 2 * 0.84;

  ctx.clearRect(0, 0, W, H);

  if (orbState === 'listening') {
    // Expanding ripple rings from orb edge
    for (let i = 0; i < 3; i++) {
      const t     = ((frame * 0.012) + i / 3) % 1;
      const ringR = r + t * 55;
      const alpha = (1 - t) * 0.65;
      ctx.beginPath();
      ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(192, 132, 252, ${alpha})`;
      ctx.lineWidth   = 2;
      ctx.stroke();
    }
  }

  if (orbState === 'speaking') {
    // Inner concentric wobble rings (audio-wave feel)
    for (let i = 0; i < 5; i++) {
      const wobble = Math.sin(frame * 0.09 + i * 1.3) * 7;
      const ringR  = r * (0.22 + i * 0.16) + wobble;
      const alpha  = 0.13 - i * 0.02;
      ctx.beginPath();
      ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
      ctx.lineWidth   = 1.5;
      ctx.stroke();
    }
  }

  frame++;
  requestAnimationFrame(drawOrb);
}
drawOrb();
