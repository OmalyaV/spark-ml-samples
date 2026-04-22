/* ============================================================
   Lyrics Genre Classifier — Application Logic
   ============================================================ */

// Genre display colours (same order as CSS custom props).
var GENRE_COLORS = {
    'pop': '#ff6b8a',
    'country': '#ffb347',
    'blues': '#5b9bd5',
    'jazz': '#b57edc',
    'reggae': '#55c67a',
    'hip hop': '#26c6da',
    'latin': '#e040fb',
    // Capitalised variants
    'Pop': '#ff6b8a',
    'Country': '#ffb347',
    'Blues': '#5b9bd5',
    'Jazz': '#b57edc',
    'Reggae': '#55c67a',
    'Rock': '#ff5252',
    'Hip Hop': '#26c6da',
    'Latin': '#e040fb'
};

var DEFAULT_COLOR = '#888888';

// ── API helpers ──────────────────────────────────────────────

function setStatus(msg, type) {
    var el = document.getElementById('status');
    el.className = 'status ' + type;
    el.innerHTML = msg;
    el.classList.remove('hidden');
}

function hideStatus() {
    document.getElementById('status').classList.add('hidden');
}

// ── Classify ─────────────────────────────────────────────────

function classifyLyrics() {
    var lyrics = document.getElementById('lyrics-input').value.trim();
    if (!lyrics) {
        setStatus('Please paste some lyrics first.', 'error');
        return;
    }

    var btn = document.getElementById('classify-btn');
    btn.disabled = true;
    setStatus('<span class="spinner"></span> Classifying...', 'info');

    fetch('/lyrics/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: lyrics
    })
        .then(function (res) {
            if (!res.ok) throw new Error('HTTP ' + res.status);
            return res.json();
        })
        .then(function (data) {
            hideStatus();
            showResults(data);
        })
        .catch(function (err) {
            setStatus('✗ Prediction failed: ' + err.message, 'error');
        })
        .finally(function () {
            btn.disabled = false;
        });
}

// ── Display Results ──────────────────────────────────────────

function showResults(data) {
    // data = { genre: "Pop", probabilities: { "pop": 0.45, "rock": 0.2, … } }
    var genre = data.genre || 'Unknown';
    var probs = data.probabilities || {};

    // Show predicted genre badge
    var badge = document.getElementById('prediction-badge');
    badge.classList.remove('hidden');
    document.getElementById('predicted-genre').textContent = genre;

    // Build sorted entries
    var entries = [];
    for (var key in probs) {
        if (probs.hasOwnProperty(key)) {
            entries.push({ genre: key, prob: probs[key] });
        }
    }
    entries.sort(function (a, b) { return b.prob - a.prob; });

    // Show charts & table
    document.getElementById('charts-container').classList.remove('hidden');
    document.getElementById('prob-table-container').classList.remove('hidden');
    document.getElementById('placeholder').classList.add('hidden');

    drawBarChart(entries);
    drawPieChart(entries, genre);
    fillTable(entries, genre);
}

// ── Bar Chart (Canvas) ──────────────────────────────────────

function drawBarChart(entries) {
    var canvas = document.getElementById('bar-chart');
    var ctx = canvas.getContext('2d');

    // High-DPI support
    var dpr = window.devicePixelRatio || 1;
    var w = 500, h = 300;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    var padding = { top: 20, right: 20, bottom: 40, left: 50 };
    var chartW = w - padding.left - padding.right;
    var chartH = h - padding.top - padding.bottom;
    var n = entries.length;
    if (n === 0) return;

    var maxVal = Math.max.apply(null, entries.map(function (e) { return e.prob; }));
    if (maxVal === 0) maxVal = 1;

    var barW = Math.min(chartW / n * 0.65, 52);
    var gap = (chartW - barW * n) / (n + 1);

    // Y-axis lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
        var yy = padding.top + chartH - (chartH * (i / 4));
        ctx.beginPath();
        ctx.moveTo(padding.left, yy);
        ctx.lineTo(w - padding.right, yy);
        ctx.stroke();

        ctx.fillStyle = '#666';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText((maxVal * i / 4 * 100).toFixed(0) + '%', padding.left - 8, yy);
    }

    // Bars
    for (var j = 0; j < n; j++) {
        var x = padding.left + gap + j * (barW + gap);
        var barH = (entries[j].prob / maxVal) * chartH;
        var y = padding.top + chartH - barH;

        var color = GENRE_COLORS[entries[j].genre] || DEFAULT_COLOR;

        // Shadow glow
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.shadowOffsetY = 4;

        // Gradient bar
        var grad = ctx.createLinearGradient(x, y, x, y + barH);
        grad.addColorStop(0, color);
        grad.addColorStop(1, hexAlpha(color, 0.5));
        ctx.fillStyle = grad;
        roundRect(ctx, x, y, barW, barH, 4);
        ctx.fill();
        ctx.restore();

        // Percentage on top
        ctx.fillStyle = '#ddd';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText((entries[j].prob * 100).toFixed(1) + '%', x + barW / 2, y - 4);

        // Label below
        ctx.fillStyle = '#aaa';
        ctx.font = '11px Inter, sans-serif';
        ctx.textBaseline = 'top';
        ctx.fillText(capitalize(entries[j].genre), x + barW / 2, padding.top + chartH + 8);
    }
}

// ── Pie Chart (Canvas) ───────────────────────────────────────

function drawPieChart(entries, predictedGenre) {
    var canvas = document.getElementById('pie-chart');
    var ctx = canvas.getContext('2d');

    var dpr = window.devicePixelRatio || 1;
    var size = 280;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, size, size);

    var cx = size / 2, cy = size / 2, r = size / 2 - 20;
    var total = 0;
    entries.forEach(function (e) { total += e.prob; });
    if (total === 0) total = 1;

    var startAngle = -Math.PI / 2;

    entries.forEach(function (e) {
        var sliceAngle = (e.prob / total) * 2 * Math.PI;
        var color = GENRE_COLORS[e.genre] || DEFAULT_COLOR;

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, startAngle, startAngle + sliceAngle);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();

        // Thin white border between slices
        ctx.strokeStyle = 'rgba(15,15,26,0.7)';
        ctx.lineWidth = 2;
        ctx.stroke();

        startAngle += sliceAngle;
    });

    // Inner circle (donut hole)
    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.52, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1a2e';
    ctx.fill();

    // Centre text
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(predictedGenre, cx, cy);

    // Legend
    var legendEl = document.getElementById('pie-legend');
    legendEl.innerHTML = '';
    entries.forEach(function (e) {
        var pct = (e.prob / total * 100).toFixed(1);
        var color = GENRE_COLORS[e.genre] || DEFAULT_COLOR;
        var item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML =
            '<span class="legend-dot" style="background:' + color + '"></span>' +
            '<span>' + capitalize(e.genre) + '</span>' +
            '<span class="legend-pct">' + pct + '%</span>';
        legendEl.appendChild(item);
    });
}

// ── Probability Table ────────────────────────────────────────

function fillTable(entries, predictedGenre) {
    var tbody = document.getElementById('prob-table-body');
    tbody.innerHTML = '';

    entries.forEach(function (e) {
        var pct = (e.prob * 100).toFixed(2);
        var isPredicted = capitalize(e.genre) === predictedGenre;
        var color = GENRE_COLORS[e.genre] || DEFAULT_COLOR;

        var tr = document.createElement('tr');
        if (isPredicted) tr.className = 'highlight';

        tr.innerHTML =
            '<td>' + (isPredicted ? '★ ' : '') + capitalize(e.genre) + '</td>' +
            '<td>' + pct + '%</td>' +
            '<td>' +
            '<div class="confidence-bar">' +
            '<div class="confidence-fill" style="width:' + pct + '%; background:' + color + ';"></div>' +
            '</div>' +
            '</td>';

        tbody.appendChild(tr);
    });
}

// ── Clear ────────────────────────────────────────────────────

function clearAll() {
    document.getElementById('lyrics-input').value = '';
    hideStatus();
    document.getElementById('prediction-badge').classList.add('hidden');
    document.getElementById('charts-container').classList.add('hidden');
    document.getElementById('prob-table-container').classList.add('hidden');
    document.getElementById('placeholder').classList.remove('hidden');
}

// ── Utilities ────────────────────────────────────────────────

function capitalize(s) {
    return s.replace(/\b\w/g, function (c) { return c.toUpperCase(); });
}

function hexAlpha(hex, alpha) {
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

function roundRect(ctx, x, y, w, h, rad) {
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.lineTo(x + w - rad, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rad);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x, y + rad);
    ctx.quadraticCurveTo(x, y, x + rad, y);
    ctx.closePath();
}
