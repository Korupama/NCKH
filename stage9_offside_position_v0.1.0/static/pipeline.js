'use strict';
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let state, chosenStage = 7, layers = new Set([2,5,6,7]), raw = false, focus = '', poseView = false;
const descriptions = [
  'Hiệu chỉnh camera và chiếu các vạch sân từ hệ tọa độ thế giới lên ảnh.',
  'Định vị con người và giữ ID qua các khung hình. Chỉ dùng quan sát đúng frame đang xem.',
  'Các điểm cơ thể và bàn chân 2D được lấy từ pose đã qua kiểm tra chất lượng.',
  'Pose 3D được đặt vào hệ tọa độ sân, rồi chiếu lại lên ảnh để quan sát độ khớp.',
  'Gán đội từ đặc trưng trang phục; màu trên ảnh thể hiện nhãn cụm đội 0 và 1.',
  'Vị trí bóng và bằng chứng tiếp xúc gần thời điểm đã chọn. Vòng vàng đánh dấu bóng.',
  'Kết hợp đội, người chạm bóng và hướng sân để tạo nhóm tấn công / đối phương.'
];
const presets = [[1],[2],[3],[4],[2,5],[6],[2,5,6,7]];
function showError(message) { $('error').hidden = false; $('error').textContent = message; }
function color(row) { return row.toucher ? '#f8eb50' : row.team_id === 0 ? '#47ade1' : row.team_id === 1 ? '#faa079' : '#a9b2bc'; }
function group(row) { return row.toucher ? 'Chạm bóng' : ({attackers:'Tấn công',opponents:'Đối phương',referees_excluded:'Trọng tài',unknown_team_excluded:'Chưa rõ đội',inactive_excluded:'Không hoạt động'})[row.group] || (row.active ? 'Chưa xác định' : 'Không có ở frame này'); }
function refreshFrame() {
  const params = new URLSearchParams({overlay:raw?'0':'1',labels:$('labels').checked?'1':'0',track:focus});
  for(let i=1;i<=7;i++) params.set('s'+i,layers.has(i)?'1':'0');
  $('frame').src = '/api/frame.jpg?'+params;
  $('raw').textContent = raw ? 'Trở lại các lớp' : 'Xem ảnh gốc';
  $('raw').setAttribute('aria-pressed',String(raw));
  $('view-label').textContent = raw ? 'Ảnh gốc từ replay' : 'Lớp đang bật: '+([...layers].sort().map(i=>'S'+i).join(' + ') || 'không có');
  document.querySelectorAll('[data-layer]').forEach(el=>{const on=layers.has(Number(el.dataset.layer));el.classList.toggle('active',on);el.setAttribute('aria-pressed',String(on));});
}
function selectStage(id) {
  chosenStage = id; layers = new Set(presets[id-1]); raw=false;
  document.querySelectorAll('.stage-card').forEach(el=>{el.classList.toggle('active',Number(el.dataset.stage)===id);el.setAttribute('aria-pressed',String(Number(el.dataset.stage)===id));});
  const s = state.stages[id-1];
  $('detail-title').textContent = 'S'+id+' · '+s.title;
  $('detail-desc').textContent = descriptions[id-1];
  $('stage-detail').textContent = s.summary;
  if(id===6) {
    const contact=state.ball.contact;
    $('stage-detail').textContent += ' · '+(contact.region||'Chưa rõ vùng chạm')+(contact.image_distance_px!=null ? ' · cách '+contact.image_distance_px.toFixed(2)+' px' : '');
  }
  if(id===7) {
    const direction=state.game_state.attack_direction;
    $('stage-detail').textContent += direction ? ' · hướng theo trục sân '+(direction.s===1?'+X':'−X') : ' · hướng chưa xác định';
  }
  $('provenance').textContent = JSON.stringify({nguon:s.source,trang_thai:s.status,chi_tiet:s.detail},null,2);
  refreshFrame();
}
function selectTrack(id) {
  focus = id;
  document.querySelectorAll('[data-row]').forEach(el=>el.classList.toggle('selected',el.dataset.row===id));
  const row=state.tracks.find(r=>r.track_id===id);
  if(!row) $('track-detail').textContent='Chọn một dòng để làm nổi bật cầu thủ và xem chất lượng pose.';
  else {
    const q=row.quality3d;
    const extra = q.reprojection_p95_px!=null ? ` · Sai lệch chiếu lại P95: ${q.reprojection_p95_px.toFixed(1)} px` : '';
    const ground = q.ground_contact_residual_cm!=null ? ` · Lệch tiếp xúc sân: ${q.ground_contact_residual_cm.toFixed(1)} cm` : '';
    $('track-detail').textContent=`${row.track_id} · ${group(row)} · 2D: ${row.pose2d_status} · 3D: ${row.pose3d_status}${extra}${ground}`;
  }
  refreshFrame(); drawPitch();
}
function drawPitch() {
  if(!state) return;
  const canvas=$('pitch'), rect=canvas.getBoundingClientRect(), ratio=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);
  const c=canvas.getContext('2d');c.scale(ratio,ratio);const w=rect.width,h=rect.height;
  c.fillStyle='#0c201d';c.fillRect(0,0,w,h);c.lineWidth=1;c.font='10px Segoe UI';
  if(poseView) { drawPose(c,w,h); return; }
  const scale=Math.min((w-54)/state.pitch.length_m,(h-38)/state.pitch.width_m);
  const xy=p=>[w/2+p[0]*scale,h/2-p[1]*scale];
  const length=state.pitch.length_m, width=state.pitch.width_m;
  for(let i=0;i<10;i++){const [x,y]=xy([-length/2+i*length/10,width/2]);c.fillStyle=i%2?'#10302a':'#102b26';c.fillRect(x,y,length/10*scale,width*scale);}
  c.strokeStyle='#42675a';
  state.pitch_lines.forEach(poly=>{c.beginPath();poly.forEach((p,i)=>{const [x,y]=xy(p);i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();});
  c.fillStyle='#8da79b';c.fillText('−X',12,h/2);c.fillText('+X',w-23,h/2);c.fillText('+Y',w/2+5,14);
  // Dots are measured root positions, not synthetic layout positions.
  state.tracks.filter(r=>r.root_world_m).forEach(row=>{const [x,y]=xy(row.root_world_m);c.globalAlpha=focus&&row.track_id!==focus?.3:1;c.fillStyle=color(row);c.beginPath();c.arc(x,y,row.track_id===focus?6:4,0,Math.PI*2);c.fill();c.strokeStyle='#071a15';c.stroke();c.fillStyle='#e3ede6';const n=Number(row.track_id.split('_').pop());c.fillText(String(n),x+6,y+(n%2?-7:12));});
  c.globalAlpha=1;
  const ball=state.ball.center_xyz_world_m;
  if(ball){const [x,y]=xy(ball);c.strokeStyle='#fff4a3';c.lineWidth=1.5;c.beginPath();c.arc(x,y,7,0,Math.PI*2);c.stroke();}
  c.fillStyle='#a3b9ae';c.fillText('Vị trí 3D · '+state.counts.pose3d+' cầu thủ có dữ liệu',16,h-9);
}
function drawPose(c,w,h) {
  const row=state.tracks.find(r=>r.track_id===focus) || state.tracks.find(r=>r.toucher&&r.root_world_m) || state.tracks.find(r=>r.root_world_m);
  if(!row?.root_world_m){c.fillStyle='#afc2b7';c.fillText('Track này không có dữ liệu pose 3D tại frame đang xem.',20,35);return;}
  const root=row.root_world_m, angle=Number($('angle').value)*Math.PI/180, scale=Math.min(w/4,h/2.8);
  const xy=p=>{const x=p[0]-root[0],y=p[1]-root[1];return [w/2+(x*Math.cos(angle)-y*Math.sin(angle))*scale,h*.8-p[2]*scale+(x*Math.sin(angle)+y*Math.cos(angle))*scale*.3];};
  c.strokeStyle='#254c40';c.lineWidth=1;
  for(let i=-2;i<=2;i++) for(const endpoints of [[[root[0]-2,root[1]+i,0],[root[0]+2,root[1]+i,0]],[[root[0]+i,root[1]-2,0],[root[0]+i,root[1]+2,0]]]) {c.beginPath();endpoints.forEach((p,j)=>{const [x,y]=xy(p);j?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();}
  c.strokeStyle=color(row);c.fillStyle=color(row);c.lineWidth=3;
  state.skeleton_edges.forEach(([a,b])=>{const pa=row.joints_world[a],pb=row.joints_world[b];if(!pa||!pb)return;c.beginPath();c.moveTo(...xy(pa));c.lineTo(...xy(pb));c.stroke();});
  row.joints_world.filter(Boolean).forEach(p=>{c.beginPath();c.arc(...xy(p),3,0,Math.PI*2);c.fill();});
  c.fillStyle='#d5e8da';c.fillText(row.track_id+' · '+group(row),16,22);
  c.fillStyle='#93ac9e';c.fillText('Mặt lưới: Z = 0 m · ô lưới 1 m · phép chiếu trực giao',16,h-12);
}
async function init() {
  try {
    const response=await fetch('/api/state');if(!response.ok)throw new Error('Không đọc được dữ liệu ('+response.status+').');
    state=await response.json();if(state.mode!=='UPSTREAM_1_7')throw new Error('Máy chủ đang chạy sai chế độ. Khởi động với --project.');
    $('frame-meta').textContent='Frame '+state.frame_index+' · '+Number(state.timestamp_sec||0).toFixed(2)+' s';
    $('source').textContent='Nguồn ảnh: '+state.frame_source.path;
    $('stages').innerHTML=state.stages.map(s=>`<button class="stage-card" data-stage="${s.id}" aria-pressed="false"><span class="number">STAGE 0${s.id}</span><strong>${escapeHtml(s.title)}</strong><span class="status ${['VALID','AVAILABLE'].includes(s.status)?'good':''}">${escapeHtml(s.status)}</span></button>`).join('');
    $('layers').innerHTML=state.stages.map(s=>`<button data-layer="${s.id}" title="${escapeHtml(s.title)}" aria-label="Lớp ${s.id}: ${escapeHtml(s.title)}" aria-pressed="false">S${s.id}</button>`).join('');
    $('stats').innerHTML=[[state.counts.detected,'Người phát hiện'],[state.counts.pose2d,'Pose 2D'],[state.counts.pose3d,'Pose 3D'],[state.counts.teams,'Được gán đội']].map(([n,label])=>`<div class="stat"><strong>${n}</strong><span>${label}</span></div>`).join('');
    const missing=state.tracks.filter(r=>r.active&&!r.root_world_m).length;
    const inactive=state.tracks.filter(r=>!r.active).length;
    $('quality-note').textContent=`Camera: ${state.stages[0].status}. ${missing} người đang quan sát thiếu pose 3D; ${inactive} track khác không có quan sát tại frame này. Vị trí bóng: ${state.ball.status}. Chưa đối chiếu độ chính xác với ground truth. Không tính đường hay kết luận việt vị.`;
    $('track-count').textContent=state.tracks.length+' track trong dữ liệu';
    $('tracks').innerHTML=state.tracks.map(row=>`<tr data-row="${escapeHtml(row.track_id)}"><td><button data-track="${escapeHtml(row.track_id)}">${escapeHtml(row.track_id.replace('track_','#'))}</button></td><td><span class="dot" style="background:${color(row)}"></span>${row.team_id==null?'—':'Đội '+escapeHtml(row.team_id)}${row.role==='goalkeeper'?' · GK':''}</td><td class="${row.pose2d.length?'ok':'missing'}">${row.pose2d.length?'Có':'Thiếu'}</td><td class="${row.root_world_m?'ok':'missing'}">${row.root_world_m?'Có':'Thiếu'}</td><td><span class="badge">${escapeHtml(group(row))}</span></td></tr>`).join('');
    document.querySelectorAll('[data-stage]').forEach(el=>el.addEventListener('click',()=>selectStage(Number(el.dataset.stage))));
    document.querySelectorAll('[data-layer]').forEach(el=>el.addEventListener('click',()=>{const id=Number(el.dataset.layer);layers.has(id)?layers.delete(id):layers.add(id);raw=false;refreshFrame();}));
    document.querySelectorAll('[data-track]').forEach(el=>el.addEventListener('click',()=>selectTrack(el.dataset.track===focus?'':el.dataset.track)));
    $('clear-track').addEventListener('click',()=>selectTrack(''));
    $('raw').addEventListener('click',()=>{raw=!raw;refreshFrame();});
    $('labels').addEventListener('change',refreshFrame);
    for(const [id,on] of [['topdown',false],['poseview',true]]) $(id).addEventListener('click',()=>{poseView=on;$('topdown').classList.toggle('active',!on);$('poseview').classList.toggle('active',on);$('rotation').hidden=!on;$('pitch-help').textContent=on?'Pose 3D của track đang chọn (mặc định: người chạm bóng). Xoay góc nhìn để quan sát; giữ nguyên sai lệch của dữ liệu gốc.':'Vị trí lấy từ root 3D. Chọn cầu thủ trong bảng để theo dõi. X dọc sân, Y ngang sân; đơn vị mét.';drawPitch();});
    $('angle').addEventListener('input',drawPitch);
    $('frame').addEventListener('error',()=>showError('Không tải được ảnh. Kiểm tra máy chủ demo rồi tải lại trang.'));
    new ResizeObserver(drawPitch).observe($('pitch'));
    selectStage(7);drawPitch();
  } catch(error) {showError(error.message);}
}
init();

// --- Live Analysis Video Player Logic ---
const videoUpload = document.getElementById('video-upload');
if (videoUpload) {
  videoUpload.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const player = document.getElementById('video-player');
    player.src = url;
    document.getElementById('video-player-wrap').style.display = 'block';
  });

  document.getElementById('video-player').addEventListener('timeupdate', function() {
    const t = this.currentTime;
    document.getElementById('video-time').textContent = t.toFixed(3);
    document.getElementById('video-frame').textContent = Math.round(t * 30);
  });

  document.getElementById('analyze-btn').addEventListener('click', async function() {
    const btn = this;
    const originalText = btn.textContent;
    btn.textContent = '�ang ph�n t�ch (vui l�ng d?i v�i ph�t)...';
    btn.disabled = true;
    btn.style.opacity = '0.7';
    
    try {
      const time = document.getElementById('video-player').currentTime;
      const res = await fetch('/api/analyze_live', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ time_sec: time, estimated_frame: Math.round(time * 30) })
      });
      
      if (!res.ok) {
        throw new Error('L?i m�y ch?: ' + await res.text());
      }
      
      await init();
      document.getElementById('stages').style.display = '';
      document.getElementById('workspace-section').style.display = '';
      document.getElementById('bottom-grid-section').style.display = '';
      alert('Ph�n t�ch ho�n t?t! Khung h�nh d� du?c c?p nh?t.');
    } catch(e) {
      alert('Chua th? ho�n th�nh: ' + e.message);
    } finally {
      btn.textContent = originalText;
      btn.disabled = false;
      btn.style.opacity = '1';
    }
  });
}


  // Override the analyze button to send image data
  document.getElementById('analyze-btn').addEventListener('click', async function(e) {
    e.stopImmediatePropagation();
    const btn = this;
    const originalText = btn.textContent;
    btn.textContent = '�ang ph�n t�ch (vui l�ng d?i v�i ph�t)...';
    btn.disabled = true;
    btn.style.opacity = '0.7';
    
    try {
      const player = document.getElementById('video-player');
      const time = player.currentTime;
      
      // Extract frame via canvas
      const canvas = document.createElement('canvas');
      canvas.width = player.videoWidth;
      canvas.height = player.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(player, 0, 0, canvas.width, canvas.height);
      const imageData = canvas.toDataURL('image/jpeg', 0.9);
      
      const res = await fetch('/api/analyze_live', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          time_sec: time, 
          estimated_frame: Math.round(time * 30),
          image: imageData
        })
      });
      
      if (!res.ok) {
        throw new Error('L?i m�y ch?: ' + await res.text());
      }
      
      await init();
      document.getElementById('stages').style.display = '';
      document.getElementById('workspace-section').style.display = '';
      document.getElementById('bottom-grid-section').style.display = '';
      alert('Ph�n t�ch ho�n t?t! Khung h�nh d� du?c c?p nh?t.');
    } catch(e) {
      alert('Chua th? ho�n th�nh: ' + e.message);
    } finally {
      btn.textContent = originalText;
      btn.disabled = false;
      btn.style.opacity = '1';
    }
  }, { capture: true });

