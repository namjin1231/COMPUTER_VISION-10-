// 낙상 감시 모니터 화면: /api/config로 카메라 목록, /stream/<k>로 영상(MJPEG), /events(SSE)로 상태·이벤트를 받는다.
// 카메라는 서버가 모두 준비되면 동시에 시작한다 (화면은 받기만 함). R = 처음부터 다시, F = 전체 화면.
const STATE_TXT = { NONE: "사람 없음", NORMAL: "정상", SUSPECT: "넘어짐 의심", DOWN: "쓰러짐 확인", ALARM: "낙상 경보" };
const tiles = [];
let lastEvN = 0, evCount = 0, gen = null, soundOn = false, audioCtx = null, beepTimer = null;

const $ = (s, r = document) => r.querySelector(s);

async function init() {
  const cfg = await fetch("/api/config").then(r => r.json());
  if (cfg.version !== 2) {                       // 예전 서버가 켜져 있으면 화면과 안 맞음
    document.body.innerHTML = `<p style="margin:40px;font-size:18px">예전 버전 서버가 실행 중입니다. 서버 창에서 Ctrl+C로 끈 뒤 monitor\\server.py를 다시 실행하세요.</p>`;
    return;
  }
  $("#site").textContent = cfg.site;
  $("#ward").textContent = cfg.ward;
  $("#sum-cams").textContent = cfg.cameras.length;
  $("#since").textContent = `시작 ${cfg.started}`;
  const j = cfg.judge;
  const rows = [["판정 모델", j.model], ["넘어짐 의심", `P ≥ ${j.th_fall} · ${j.n_fall}프레임`], ["속도 조건", `≥ ${j.v_gate}`],
    ["쓰러짐 확인", `P ≥ ${j.th_down} · ${j.n_down}프레임 (${j.window_s}초 안)`], ["경보 대기", `${cfg.alarm_s}초 (현장 권장 5초)`]];
  $("#cfg").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

  const wall = $("#wall"), tpl = $("#tile-tpl"), list = $("#status-list");
  cfg.cameras.forEach((c, k) => {
    const el = tpl.content.firstElementChild.cloneNode(true);
    $(".cam-id", el).textContent = c.id;
    $(".cam-name", el).textContent = c.name;
    const img = $(".feed", el);
    img.alt = `${c.name} 카메라 화면`;
    img.src = `/stream/${k}`;
    el.addEventListener("click", e => { if (!e.target.closest(".ack-btn")) toggleFocus(k); });
    el.addEventListener("keydown", e => { if (e.key === "Enter") toggleFocus(k); });
    $(".ack-btn", el).addEventListener("click", () => ack(k));
    wall.appendChild(el);
    tiles.push(el);
    const li = document.createElement("li");
    li.innerHTML = `<span class="sdot"></span><span class="sname">${c.name}<small>${c.id}</small></span><span class="sstate">연결 중</span>`;
    li.addEventListener("click", () => toggleFocus(k));
    list.appendChild(li);
  });
  connect();
}

function toggleFocus(k) {
  const wall = $("#wall"), on = !(wall.classList.contains("focus") && tiles[k].classList.contains("focused"));
  tiles.forEach(t => t.classList.remove("focused"));
  wall.classList.toggle("focus", on);
  if (on) tiles[k].classList.add("focused");
}

function connect() {
  const es = new EventSource("/events");
  es.onopen = () => { $(".live").classList.add("ok"); $("#conn").textContent = "LIVE"; };
  es.onerror = () => { $(".live").classList.remove("ok"); $("#conn").textContent = "재연결 중"; };
  es.onmessage = m => update(JSON.parse(m.data));
}

function update(d) {
  if (gen !== d.gen) { gen = d.gen; resetEvents(); }      // 처음부터 다시 하면 이벤트 기록도 새로
  let nOk = 0, nWarn = 0, nAlarm = 0;
  const items = $("#status-list").children;
  d.cams.forEach((c, k) => {
    const el = tiles[k]; if (!el) return;
    el.dataset.state = c.state;
    el.classList.toggle("alarm", c.alarm);
    if (c.status !== "연결 중") el.classList.add("ready");
    $(".state-txt", el).textContent = c.alarm && c.state !== "ALARM" ? "경보 (확인 대기)" : STATE_TXT[c.state];
    $(".pf b", el).textContent = c.p[1].toFixed(2);
    $(".pd b", el).textContent = c.p[2].toFixed(2);
    $(".fps", el).textContent = `${c.fps.toFixed(1)} fps`;
    $(".veil-txt", el).textContent = c.status;
    $(".alarm-where", el).textContent = `${c.name} · ${c.id} · ${c.alarm_since || ""}`;
    const li = items[k];
    li.dataset.state = c.state; li.classList.toggle("alarm", c.alarm);
    $(".sstate", li).textContent = c.alarm ? "낙상 경보" : (c.status === "LIVE" || c.status === "다음 영상 준비" ? STATE_TXT[c.state] : c.status);
    if (c.alarm) nAlarm++; else if (c.state === "SUSPECT" || c.state === "DOWN") nWarn++; else nOk++;
  });
  $("#sum-ok").textContent = nOk; $("#sum-warn").textContent = nWarn; $("#sum-alarm").textContent = nAlarm;
  $(".summary").classList.toggle("has-alarm", nAlarm > 0);
  document.title = nAlarm ? `● 경보 ${nAlarm} — 낙상 감시 모니터` : "낙상 감시 모니터";
  setBeep(nAlarm > 0);
  if (d.events && d.events.length) addEvents(d.events);
}

function resetEvents() {
  lastEvN = 0; evCount = 0;
  $("#events").innerHTML = `<li class="empty">아직 이벤트가 없습니다</li>`;
  $("#ev-count").textContent = 0;
}

function addEvents(evs) {
  const ol = $("#events"), empty = $(".empty", ol);
  if (empty) empty.remove();
  evs.forEach(e => {
    if (e.n <= lastEvN) return; lastEvN = e.n; evCount++;
    const li = document.createElement("li");
    li.dataset.kind = e.kind;
    li.innerHTML = `<span class="ev-time">${e.time}</span><div class="ev-main"><span class="ev-title"></span><span class="ev-where"></span></div>`;
    $(".ev-title", li).textContent = e.text;
    $(".ev-where", li).textContent = `${e.name} · ${e.cam_id}`;
    li.title = `영상: ${e.clip}`;
    ol.prepend(li);
    while (ol.children.length > 120) ol.lastChild.remove();
  });
  $("#ev-count").textContent = evCount;
}

function ack(k) { fetch(`/api/ack/${k}`, { method: "POST" }); }

// 경보음 (브라우저 정책상 버튼을 한 번 눌러야 소리가 남)
$("#sound-btn").addEventListener("click", () => {
  soundOn = !soundOn;
  if (soundOn && !audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  $("#sound-btn").classList.toggle("on", soundOn);
  $("#sound-txt").textContent = soundOn ? "경보음 켜짐" : "경보음 꺼짐";
});
function setBeep(on) {
  if (on && soundOn && !beepTimer) { beep(); beepTimer = setInterval(beep, 1200); }
  if ((!on || !soundOn) && beepTimer) { clearInterval(beepTimer); beepTimer = null; }
}
function beep() {
  if (!audioCtx) return;
  [0, 0.18].forEach(dt => {
    const o = audioCtx.createOscillator(), g = audioCtx.createGain(), t = audioCtx.currentTime + dt;
    o.type = "sine"; o.frequency.value = 880; o.connect(g); g.connect(audioCtx.destination);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.25, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);
    o.start(t); o.stop(t + 0.15);
  });
}

// 시계
function tick() {
  const n = new Date(), p = v => String(v).padStart(2, "0");
  $("#clock-time").textContent = `${p(n.getHours())}:${p(n.getMinutes())}:${p(n.getSeconds())}`;
  $("#clock-date").textContent = n.toLocaleDateString("ko-KR", { year: "numeric", month: "long", day: "numeric", weekday: "short" });
}
setInterval(tick, 1000); tick();

// 키보드: 1~4 크게 보기, Esc 돌아가기, R 처음부터 다시(모든 카메라 동시), F 전체 화면
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { $("#wall").classList.remove("focus"); tiles.forEach(t => t.classList.remove("focused")); }
  else if (/^[1-9]$/.test(e.key) && tiles[+e.key - 1]) toggleFocus(+e.key - 1);
  else if (e.key === "r" || e.key === "R") fetch("/api/restart", { method: "POST" });
  else if (e.key === "f" || e.key === "F") { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
});
init();
