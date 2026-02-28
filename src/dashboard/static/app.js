/**
 * VayuSwarm — Mission Control Dashboard
 * Real-time WebSocket client + tactical map rendering
 */

// ─── State ────────────────────────────────────────────────────
let ws = null;
let startTime = Date.now();
let drones = {};
let events = [];
let commands = [];

// ─── WebSocket Connection ─────────────────────────────────────
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws`;

    ws = new WebSocket(url);

    ws.onopen = () => {
        document.querySelector('#conn-status .indicator-dot').style.background = 'var(--green)';
        document.querySelector('#conn-status span').textContent = 'CONNECTED';
        console.log('[VayuSwarm] WebSocket connected');
    };

    ws.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            if (data.type === 'update') {
                handleUpdate(data);
            }
        } catch (err) {
            console.error('[VayuSwarm] Parse error:', err);
        }
    };

    ws.onclose = () => {
        document.querySelector('#conn-status .indicator-dot').style.background = 'var(--red)';
        document.querySelector('#conn-status span').textContent = 'DISCONNECTED';
        console.log('[VayuSwarm] WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('[VayuSwarm] WebSocket error:', err);
    };
}

// ─── Handle Update ────────────────────────────────────────────
function handleUpdate(data) {
    // Fleet
    if (data.fleet) {
        updateDroneCards(data.fleet);
        updateMap(data.fleet);
    }

    // Events
    if (data.events) {
        updateEventFeed(data.events);
    }

    // Commands
    if (data.commands) {
        updateCommandLog(data.commands);
    }

    // Mission
    if (data.mission) {
        updateMission(data.mission);
    }

    // Stats
    if (data.stats) {
        updateStats(data.stats);
    }

    // Header stats
    const fleet = data.fleet || [];
    const online = fleet.filter(d => d.online).length;
    document.getElementById('drone-count').textContent = `${online}/${fleet.length}`;
    document.getElementById('fleet-badge').textContent = `${online} active`;

    // Threat count
    let threats = 0;
    (data.events || []).forEach(e => {
        if (e.threat_level === 'CRITICAL' || e.threat_level === 'HIGH') threats++;
    });
    document.getElementById('threat-count').textContent = threats;
}

// ─── Drone Cards ──────────────────────────────────────────────
function updateDroneCards(fleet) {
    const container = document.getElementById('drone-cards');

    fleet.forEach(drone => {
        let card = document.getElementById(`drone-${drone.drone_id}`);

        if (!card) {
            card = document.createElement('div');
            card.id = `drone-${drone.drone_id}`;
            card.className = 'drone-card';
            container.appendChild(card);
        }

        const state = drone.state || 'IDLE';
        const battery = drone.battery || 0;
        const batteryClass = battery > 50 ? 'high' : battery > 25 ? 'mid' : 'low';

        card.className = `drone-card state-${state}`;
        card.innerHTML = `
            <div class="drone-card-header">
                <span class="drone-id">${drone.drone_id}</span>
                <span class="drone-state ${state}">${state}</span>
            </div>
            <div class="drone-metrics">
                <div class="metric">BAT <span class="val">${battery.toFixed(0)}%</span></div>
                <div class="metric">SPD <span class="val">${(drone.speed || 0).toFixed(1)}m/s</span></div>
                <div class="metric">ALT <span class="val">${drone.position ? drone.position.alt.toFixed(0) : 0}m</span></div>
                <div class="metric">SIG <span class="val">${(drone.signal || 0).toFixed(0)}%</span></div>
            </div>
            <div class="battery-bar">
                <div class="battery-fill ${batteryClass}" style="width: ${battery}%"></div>
            </div>
            ${drone.warnings && drone.warnings.length > 0 ?
                `<div style="margin-top:6px;font-size:9px;color:var(--red);font-family:var(--font-mono)">
                    ⚠ ${drone.warnings.join(', ')}
                </div>` : ''}
        `;
    });
}

// ─── Tactical Map ─────────────────────────────────────────────
function updateMap(fleet) {
    const canvas = document.getElementById('tactical-map');
    if (!canvas) return;

    // Resize canvas to container
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight - 60; // Header + legend

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    // Clear
    ctx.fillStyle = '#0a0e17';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Find bounds from drone positions
    const positions = fleet
        .filter(d => d.position)
        .map(d => d.position);

    if (positions.length === 0) {
        // No drones — show placeholder
        ctx.fillStyle = 'rgba(255,255,255,0.1)';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for drone telemetry...', w / 2, h / 2);
        return;
    }

    // Compute map bounds
    let minLat = Infinity, maxLat = -Infinity;
    let minLon = Infinity, maxLon = -Infinity;
    positions.forEach(p => {
        minLat = Math.min(minLat, p.lat);
        maxLat = Math.max(maxLat, p.lat);
        minLon = Math.min(minLon, p.lon);
        maxLon = Math.max(maxLon, p.lon);
    });

    // Add padding
    const padLat = Math.max(0.002, (maxLat - minLat) * 0.3);
    const padLon = Math.max(0.002, (maxLon - minLon) * 0.3);
    minLat -= padLat; maxLat += padLat;
    minLon -= padLon; maxLon += padLon;

    const toX = lon => ((lon - minLon) / (maxLon - minLon)) * (w - 40) + 20;
    const toY = lat => ((maxLat - lat) / (maxLat - minLat)) * (h - 40) + 20;

    // Draw drones
    fleet.forEach(drone => {
        if (!drone.position) return;

        const x = toX(drone.position.lon);
        const y = toY(drone.position.lat);
        const state = drone.state || 'IDLE';

        // Color by state
        let color = '#5a6478';
        if (state === 'PATROL') color = '#00e676';
        else if (state === 'INVESTIGATE') color = '#ffc107';
        else if (state === 'TRACK') color = '#ff9100';
        else if (state === 'EMERGENCY' || state === 'RTL') color = '#ff1744';

        // Glow
        ctx.beginPath();
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
        gradient.addColorStop(0, color + '40');
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.arc(x, y, 20, 0, Math.PI * 2);
        ctx.fill();

        // Drone marker
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();

        // Heading indicator
        const rad = (drone.heading || 0) * Math.PI / 180 - Math.PI / 2;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(rad) * 15, y + Math.sin(rad) * 15);
        ctx.stroke();

        // Label
        ctx.font = '10px JetBrains Mono';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.fillText(drone.drone_id, x, y - 12);
    });
}

// ─── Event Feed ───────────────────────────────────────────────
function updateEventFeed(newEvents) {
    const list = document.getElementById('event-list');
    document.getElementById('event-badge').textContent = `${newEvents.length} events`;

    list.innerHTML = '';
    newEvents.slice().reverse().forEach(event => {
        const div = document.createElement('div');
        div.className = 'event-item';

        const time = new Date(event.timestamp * 1000);
        const timeStr = time.toLocaleTimeString('en-US', { hour12: false });

        div.innerHTML = `
            <span class="event-time">${timeStr}</span>
            <span class="event-drone">${event.drone_id}</span>
            <span class="event-text">${event.event}</span>
            <span class="threat-tag threat-${event.threat_level}">${event.threat_level}</span>
        `;
        list.appendChild(div);
    });
}

// ─── Command Log ──────────────────────────────────────────────
function updateCommandLog(cmds) {
    const list = document.getElementById('command-list');
    list.innerHTML = '';

    cmds.slice().reverse().forEach(cmd => {
        const div = document.createElement('div');
        div.className = 'command-item';
        div.innerHTML = `
            <span class="command-drone">${cmd.drone_id}</span>
            <span class="command-type">${cmd.command}</span>
            <span class="command-msg">${cmd.message || ''}</span>
        `;
        list.appendChild(div);
    });
}

// ─── Mission ──────────────────────────────────────────────────
function updateMission(mission) {
    document.getElementById('mission-name').textContent = mission.name || 'No active mission';
    const progress = document.getElementById('mission-progress');
    progress.style.width = `${mission.progress_pct || 0}%`;
    document.getElementById('mission-stats').textContent =
        `${mission.objectives_completed || 0}/${mission.objectives_total || 0} objectives • ${mission.drones || 0} drones`;
}

// ─── Stats ────────────────────────────────────────────────────
function updateStats(stats) {
    document.getElementById('stat-decisions').textContent = stats.llm?.decisions || 0;
    document.getElementById('stat-events').textContent = stats.events_logged || 0;
    document.getElementById('stat-commands').textContent = stats.commands_sent || 0;
    document.getElementById('cycle-count').textContent = stats.cycles || 0;
}

// ─── Uptime Timer ─────────────────────────────────────────────
function updateUptime() {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    document.getElementById('uptime').textContent = `${h}:${m}:${s}`;
}

// ─── Polling Fallback ─────────────────────────────────────────
async function pollData() {
    try {
        const [fleet, events, cmds, mission, stats] = await Promise.all([
            fetch('/api/fleet').then(r => r.json()),
            fetch('/api/events').then(r => r.json()),
            fetch('/api/commands').then(r => r.json()),
            fetch('/api/mission').then(r => r.json()),
            fetch('/api/stats').then(r => r.json()),
        ]);

        handleUpdate({
            type: 'update',
            fleet: fleet.drones || [],
            events: events.events || [],
            commands: cmds.commands || [],
            mission: mission.progress || {},
            stats: stats,
        });
    } catch (err) {
        // API not available yet
    }
}

// ─── Initialize ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    setInterval(updateUptime, 1000);
    // Poll as fallback every 2s
    setInterval(pollData, 2000);
    // Initial poll
    pollData();
});
