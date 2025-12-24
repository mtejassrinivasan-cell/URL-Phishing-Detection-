const form = document.getElementById('checkForm');
const urlInput = document.getElementById('urlInput');
const resultEl = document.getElementById('result');
const analyzeBtn = document.getElementById('analyzeBtn');
const themeToggle = document.getElementById('themeToggle');
const resetLink = document.getElementById('resetLink');

function setResult({ verdict, probability, message }) {
  resultEl.className = 'result';
  resultEl.classList.add(verdict === 'Legitimate' ? 'success' : 'danger');
  resultEl.innerHTML = `
    <strong>${verdict}</strong>
    <div>Probability: ${(probability * 100).toFixed(1)}%</div>
    <div>${message}</div>
  `;
  resultEl.classList.remove('hidden');
}

function isLikelyUrl(value) {
  try {
    const u = new URL(value);
    return Boolean(u.protocol && u.hostname);
  } catch {
    return false;
  }
}

async function apiPredict(url) {
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || 'Request failed');
  }
  return res.json();
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const value = urlInput.value.trim();
  resultEl.classList.add('hidden');

  if (!value || !isLikelyUrl(value)) {
    setResult({ verdict: 'Phishing', probability: 0.9, message: 'Please enter a valid URL like https://example.com' });
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Checking…';

  try {
    const data = await apiPredict(value);
    setResult(data);
  } catch (err) {
    setResult({ verdict: 'Phishing', probability: 0.8, message: 'Something went wrong. Please try again.' });
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Check';
  }
});

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.dataset.theme;
  document.documentElement.dataset.theme = current === 'light' ? 'dark' : 'light';
});

resetLink.addEventListener('click', (e) => {
  e.preventDefault();
  urlInput.value = '';
  resultEl.classList.add('hidden');
  urlInput.focus();
});


