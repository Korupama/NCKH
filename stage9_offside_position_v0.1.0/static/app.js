const $ = (id) => document.getElementById(id);

function fmt(value, digits=2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function frameUrl() {
  const qs = new URLSearchParams({
    overlay: $('toggleOverlay').checked ? '1':'0',
    reference: $('toggleRef').checked ? '1':'0',
    defenders: $('toggleDef').checked ? '1':'0',
    all_defenders: $('toggleAllDef').checked ? '1':'0',
    skeleton: $('toggleSkeleton').checked ? '1':'0',
    labels: $('toggleLabels').checked ? '1':'0',
    t: Date.now().toString()
  });
  return '/api/frame.jpg?' + qs.toString();
}

function refreshFrame(){ $('frameImage').src = frameUrl(); }

function badgeClass(label){
  if(label === 'OFFSIDE_POSITION') return 'off';
  if(label === 'ONSIDE') return 'on';
  if(label === 'TOUCHER_EXCLUDED') return 'passer';
  return 'unknown';
}

function badgeText(row){
  if(row.label === 'OFFSIDE_POSITION') return 'OFFSIDE';
  if(row.label === 'ONSIDE') return 'ONSIDE';
  if(row.label === 'TOUCHER_EXCLUDED') return 'PASSER';
  return 'ON?';
}

function attackerRow(row){
  const delta = row.delta_q_m;
  const deltaClass = delta !== null && delta !== undefined && Number(delta) > 0 ? 'positive':'negative';
  const source = row.geometry?.source || 'no metric geometry';
  return `<div class="player-row">
    <span class="badge ${badgeClass(row.label)}">${badgeText(row)}</span>
    <div class="player-main"><div class="player-id">${row.track_id}</div><div class="player-meta">${source} · ${row.reason || ''}</div></div>
    <div class="player-delta ${deltaClass}">${delta === null || delta === undefined ? '—' : (Number(delta)>=0?'+':'')+fmt(delta)+' m'}</div>
  </div>`;
}

function opponentRow(row){
  return `<div class="player-row">
    <span class="badge def">${row.second_last ? '2ND LAST' : (row.rank ? '#'+row.rank : 'DEF')}</span>
    <div class="player-main"><div class="player-id">${row.track_id}</div><div class="player-meta">${row.anchor?.name || 'geometry unavailable'}</div></div>
    <div class="player-delta">${fmt(row.goalward_q_m)} m</div>
  </div>`;
}

async function init(){
  const state = await (await fetch('/api/state')).json();
  $('modePill').textContent = state.mode || state.status || 'DEMO';
  $('frameMetric').textContent = state.frame_index ?? '—';
  $('directionMetric').textContent = state.attack_direction?.label || ('s='+(state.attack_direction?.s ?? '—'));
  $('referenceMetric').textContent = state.reference?.X_world_m === null || state.reference?.X_world_m === undefined ? '—' : fmt(state.reference.X_world_m)+' m';
  $('referenceSourceMetric').textContent = state.reference?.source || '—';
  const fs = state.visualization?.frame_source || {};
  $('frameSourceMetric').textContent = fs.path ? `${fs.origin || fs.kind}: ${fs.path}` : (fs.kind || '—');
  const sw = $('sourceWarning');
  if (fs.warning) { sw.textContent = fs.warning + ' — pass a clean raw replay with --video or --image if overlays are baked into the source.'; sw.hidden = false; } else { sw.hidden = true; }
  const off = (state.attackers||[]).filter(x=>x.label==='OFFSIDE_POSITION').length;
  const on = (state.attackers||[]).filter(x=>x.label==='ONSIDE').length;
  const passer = (state.attackers||[]).filter(x=>x.label==='TOUCHER_EXCLUDED').length;
  $('attackerSummary').textContent = `${off} offside · ${on} onside · ${passer} passer`;
  $('attackerList').innerHTML = (state.attackers||[]).length ? state.attackers.map(attackerRow).join('') : '<div class="empty">No attackers in Stage 7 context.</div>';
  $('opponentList').innerHTML = (state.opponents||[]).length ? state.opponents.map(opponentRow).join('') : '<div class="empty">No opponent ranking available.</div>';
  refreshFrame();
}

['toggleOverlay','toggleRef','toggleDef','toggleAllDef','toggleSkeleton','toggleLabels'].forEach(id => $(id).addEventListener('change', refreshFrame));
init().catch(err => { document.body.innerHTML += `<pre>${err}</pre>`; });
