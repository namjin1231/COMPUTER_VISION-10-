import csv
import importlib
import json
import os
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO

from common import FPS, LOGS, ROOT
from demo_video import (
    LOST_DROP,
    MERGE_DIST,
    MERGE_GAP,
    SEVERITY,
    Person,
    box_iou,
    draw,
    load_models,
    predict,
)
from features import compute_features

extract_mod = importlib.import_module("03_extract_pose")


class FallMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("낙상 감지 모니터")
        self.root.geometry("1280x780")
        self.root.minsize(1060, 680)
        self.root.configure(bg="#FFFFFF")

        self.is_running = False
        self.is_paused = False
        self.show_settings = False
        self.worker_thread = None
        self.current_photo = None

        # 로딩 오버레이 상태 변수
        self.is_loading = False
        self.loading_angle = 0
        self.loading_subtext = "잠시만 기다려 주세요..."
        self.loading_after_id = None

        # 그래프 및 로그 상태 추적용
        self.prob_history = deque(maxlen=int(10 * FPS))  # 최근 10초 (150프레임)
        self.tracked_graph_id = None
        self.prev_person_states = {}
        self.alarm_triggered_ids = set()

        self._init_variables()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _resolve_default_paths(self):
        parent_model_dir = ROOT.parent / "1_모델"
        if parent_model_dir.exists():
            m_paths = [
                str((parent_model_dir / f"tcnte_seed{s}" / "best.pt").resolve())
                for s in (0, 1, 2)
            ]
            w_path = str((parent_model_dir / "yolo26n-pose.pt").resolve())
        else:
            m_paths = [
                str((ROOT / f"runs/Rb_r2_sel/tcnte_seed{s}/best.pt").resolve())
                for s in (0, 1, 2)
            ]
            w_path = str((ROOT / "yolo26n-pose.pt").resolve())
        return " ; ".join(m_paths), w_path

    def _init_variables(self):
        default_models, default_weights = self._resolve_default_paths()

        self.var_source_type = tk.StringVar(value="video")
        self.var_video_path = tk.StringVar(value="")
        self.var_webcam_idx = tk.IntVar(value=0)
        self.var_camera_name = tk.StringVar(value="거실 카메라")

        self.var_models = tk.StringVar(value=default_models)
        self.var_weights = tk.StringVar(value=default_weights)

        self.var_preset = tk.StringVar(value="final")
        self.var_alarm_s = tk.DoubleVar(value=2.0)
        self.var_th_fall = tk.DoubleVar(value=0.5)
        self.var_th_down = tk.DoubleVar(value=0.4)
        self.var_n_fall = tk.IntVar(value=2)
        self.var_n_down = tk.IntVar(value=4)
        self.var_window_s = tk.DoubleVar(value=5.0)
        self.var_smooth = tk.IntVar(value=3)
        self.var_v_gate = tk.DoubleVar(value=0.02)
        self.var_use_v_gate = tk.BooleanVar(value=True)

        self.var_stitch = tk.BooleanVar(value=True)
        self.var_single_person = tk.BooleanVar(value=True)
        self.var_latch_alarm = tk.BooleanVar(value=False)
        self.var_fix_release = tk.BooleanVar(value=True)
        self.var_save_output = tk.BooleanVar(value=True)
        self.var_sound_alarm = tk.BooleanVar(value=True)

        self.var_progress = tk.DoubleVar(value=0.0)
        self.var_progress_text = tk.StringVar(value="대기 중 (0.0%)")

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor="#E2E8F0",
            background="#2563EB",
            bordercolor="#CBD5E1",
            lightcolor="#3B82F6",
            darkcolor="#1D4ED8",
            thickness=14,
        )

        # ==================== 1. 최상단 다크 헤더 바 ====================
        header = tk.Frame(self.root, bg="#1E242D", height=54)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        tk.Label(
            header,
            text="낙상 감지 모니터",
            bg="#1E242D",
            fg="#FFFFFF",
            font=("맑은 고딕", 14, "bold"),
        ).pack(side=tk.LEFT, padx=20)

        self.lbl_header_status = tk.Label(
            header,
            text="입력: 대기 중 | 0.0 FPS | 0 ms",
            bg="#1E242D",
            fg="#E2E8F0",
            font=("맑은 고딕", 11),
        )
        self.lbl_header_status.pack(side=tk.RIGHT, padx=20)

        # ==================== 2. 컴팩트 실행/소스 컨트롤 바 ====================
        toolbar = tk.Frame(self.root, bg="#F8FAFC", bd=0, highlightbackground="#E2E8F0", highlightthickness=1)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        tb_inner = tk.Frame(toolbar, bg="#F8FAFC", padx=16, pady=7)
        tb_inner.pack(fill=tk.X)

        tk.Radiobutton(
            tb_inner,
            text="동영상",
            variable=self.var_source_type,
            value="video",
            command=self._toggle_source,
            bg="#F8FAFC",
            font=("맑은 고딕", 9, "bold"),
        ).pack(side=tk.LEFT)

        self.ent_video = ttk.Entry(tb_inner, textvariable=self.var_video_path, width=36)
        self.ent_video.pack(side=tk.LEFT, padx=(4, 4))

        self.btn_browse_vid = ttk.Button(tb_inner, text="파일 선택", width=9, command=self._browse_video)
        self.btn_browse_vid.pack(side=tk.LEFT, padx=(0, 12))

        tk.Radiobutton(
            tb_inner,
            text="웹캠",
            variable=self.var_source_type,
            value="webcam",
            command=self._toggle_source,
            bg="#F8FAFC",
            font=("맑은 고딕", 9, "bold"),
        ).pack(side=tk.LEFT)

        self.spn_cam = ttk.Spinbox(tb_inner, from_=0, to=9, textvariable=self.var_webcam_idx, width=3, state="disabled")
        self.spn_cam.pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(tb_inner, text="위치명:", bg="#F8FAFC", font=("맑은 고딕", 9)).pack(side=tk.LEFT)
        ttk.Entry(tb_inner, textvariable=self.var_camera_name, width=11).pack(side=tk.LEFT, padx=(4, 14))

        # 우측 제어 버튼들
        self.btn_toggle_cfg = ttk.Button(tb_inner, text="상세 설정 ▾", width=11, command=self._toggle_settings_drawer)
        self.btn_toggle_cfg.pack(side=tk.RIGHT, padx=(6, 0))

        self.btn_stop = ttk.Button(tb_inner, text="⏹ 정지", width=8, command=self.stop_detection, state="disabled")
        self.btn_stop.pack(side=tk.RIGHT, padx=3)

        self.btn_pause = ttk.Button(tb_inner, text="⏸ 일시정지", width=10, command=self.toggle_pause, state="disabled")
        self.btn_pause.pack(side=tk.RIGHT, padx=3)

        self.btn_start = tk.Button(
            tb_inner,
            text="▶ 감지 시작",
            bg="#1E242D",
            fg="#FFFFFF",
            activebackground="#334155",
            activeforeground="#FFFFFF",
            font=("맑은 고딕", 9, "bold"),
            relief=tk.FLAT,
            padx=14,
            pady=3,
            cursor="hand2",
            command=self.start_detection,
        )
        self.btn_start.pack(side=tk.RIGHT, padx=(0, 4))

        # ==================== 2-1. 접이식 상세 설정 패널 (기본 숨김) ====================
        self.settings_drawer = tk.Frame(self.root, bg="#F1F5F9", highlightbackground="#CBD5E1", highlightthickness=1)
        self._build_settings_drawer()

        # ==================== 3. 메인 대시보드 본문 (좌: 영상 / 우: 경보·그래프·로그) ====================
        body = tk.Frame(self.root, bg="#FFFFFF", padx=16, pady=14)
        body.pack(fill=tk.BOTH, expand=True)

        body.columnconfigure(0, weight=65)
        body.columnconfigure(1, weight=35, minsize=340)
        body.rowconfigure(0, weight=1)

        # -------------------- 좌측 영역: 영상 오버레이 + 진행상황 게이지 --------------------
        left_col = tk.Frame(body, bg="#FFFFFF")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left_col.rowconfigure(0, weight=1)
        left_col.columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(left_col, bg="#232933", highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self.video_canvas.bind("<Configure>", self._on_video_canvas_resize)

        # 좌측 상단 오버레이 안내 문구
        self.overlay_hint_id = self.video_canvas.create_text(
            18,
            16,
            text="영상 + 관절 오버레이",
            fill="#CBD5E1",
            font=("맑은 고딕", 11, "bold"),
            anchor=tk.NW,
        )
        self.center_hint_id = self.video_canvas.create_text(
            380,
            240,
            text="상단에서 동영상 파일 또는 웹캠을 선택한 뒤 [▶ 감지 시작]을 누르세요.",
            fill="#94A3B8",
            font=("맑은 고딕", 11),
            anchor=tk.CENTER,
        )

        # 영상 하단 진행상황 게이지 영역
        prog_card = tk.Frame(
            left_col, bg="#F8FAFC", highlightbackground="#E2E8F0", highlightthickness=1, padx=12, pady=8
        )
        prog_card.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        prog_top = tk.Frame(prog_card, bg="#F8FAFC")
        prog_top.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            prog_top, text="영상 분석 진행상황", bg="#F8FAFC", fg="#334155", font=("맑은 고딕", 9, "bold")
        ).pack(side=tk.LEFT)

        tk.Label(
            prog_top,
            textvariable=self.var_progress_text,
            bg="#F8FAFC",
            fg="#1E293B",
            font=("맑은 고딕", 9, "bold"),
        ).pack(side=tk.RIGHT)

        self.progressbar = ttk.Progressbar(
            prog_card,
            variable=self.var_progress,
            maximum=100.0,
            mode="determinate",
            style="Custom.Horizontal.TProgressbar",
        )
        self.progressbar.pack(fill=tk.X)

        # -------------------- 우측 영역: 3단 카드 스택 --------------------
        right_col = tk.Frame(body, bg="#FFFFFF")
        right_col.grid(row=0, column=1, sticky="nsew")
        right_col.columnconfigure(0, weight=1)
        right_col.rowconfigure(0, weight=0)
        right_col.rowconfigure(1, weight=1)
        right_col.rowconfigure(2, weight=1)

        # [카드 1] 상단 상태/경보 배너 카드
        self.alert_card = tk.Frame(right_col, bg="#1E293B", height=112, padx=16, pady=14)
        self.alert_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.alert_card.pack_propagate(False)

        self.lbl_alert_title = tk.Label(
            self.alert_card,
            text="감시 대기 중",
            bg="#1E293B",
            fg="#FFFFFF",
            font=("맑은 고딕", 17, "bold"),
        )
        self.lbl_alert_title.pack(expand=True, pady=(4, 0))

        self.lbl_alert_sub = tk.Label(
            self.alert_card,
            text=f"--:--:-- · {self.var_camera_name.get()}",
            bg="#1E293B",
            fg="#E2E8F0",
            font=("맑은 고딕", 10),
        )
        self.lbl_alert_sub.pack(expand=True, pady=(0, 4))

        # [카드 2] 중앙 낙상 확률 그래프 카드 (최근 10초)
        graph_card = tk.Frame(
            right_col, bg="#F1F5F9", highlightbackground="#CBD5E1", highlightthickness=1, padx=12, pady=10
        )
        graph_card.grid(row=1, column=0, sticky="nsew", pady=(0, 12))

        self.lbl_graph_title = tk.Label(
            graph_card,
            text="ID - 낙상 확률 (최근 10초)",
            bg="#F1F5F9",
            fg="#1E293B",
            font=("맑은 고딕", 10, "bold"),
            anchor=tk.W,
        )
        self.lbl_graph_title.pack(fill=tk.X, pady=(0, 4))

        self.graph_canvas = tk.Canvas(graph_card, bg="#F1F5F9", highlightthickness=0, height=160)
        self.graph_canvas.pack(fill=tk.BOTH, expand=True)
        self.graph_canvas.bind("<Configure>", lambda _e: self._draw_prob_graph())

        # [카드 3] 하단 경보 로그 카드
        log_card = tk.Frame(
            right_col, bg="#F1F5F9", highlightbackground="#CBD5E1", highlightthickness=1, padx=12, pady=10
        )
        log_card.grid(row=2, column=0, sticky="nsew")

        tk.Label(
            log_card,
            text="경보 로그",
            bg="#F1F5F9",
            fg="#334155",
            font=("맑은 고딕", 10, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 6))

        log_inner = tk.Frame(log_card, bg="#F1F5F9")
        log_inner.pack(fill=tk.BOTH, expand=True)

        self.txt_log = tk.Text(
            log_inner,
            bg="#F1F5F9",
            fg="#0F172A",
            font=("맑은 고딕", 10),
            relief=tk.FLAT,
            wrap=tk.WORD,
            state="disabled",
            height=7,
        )
        scroll = ttk.Scrollbar(log_inner, orient=tk.VERTICAL, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=scroll.set)
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.root.after(100, self._draw_prob_graph)

    def _build_settings_drawer(self):
        inner = tk.Frame(self.settings_drawer, bg="#F1F5F9", padx=16, pady=10)
        inner.pack(fill=tk.X)

        r1 = tk.Frame(inner, bg="#F1F5F9")
        r1.pack(fill=tk.X, pady=2)
        tk.Label(r1, text="TCNTE 모델:", bg="#F1F5F9", font=("맑은 고딕", 9, "bold"), width=11, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_models, width=52).pack(side=tk.LEFT, padx=4)
        ttk.Button(r1, text="찾기", width=6, command=self._browse_models).pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(r1, text="YOLO Pose:", bg="#F1F5F9", font=("맑은 고딕", 9, "bold")).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_weights, width=32).pack(side=tk.LEFT, padx=4)
        ttk.Button(r1, text="찾기", width=6, command=self._browse_weights).pack(side=tk.LEFT)

        r2 = tk.Frame(inner, bg="#F1F5F9")
        r2.pack(fill=tk.X, pady=(6, 2))
        tk.Label(r2, text="프리셋:", bg="#F1F5F9", font=("맑은 고딕", 9, "bold"), width=11, anchor=tk.W).pack(side=tk.LEFT)
        cb = ttk.Combobox(r2, textvariable=self.var_preset, values=["final", "B", "v1"], state="readonly", width=7)
        cb.pack(side=tk.LEFT, padx=(4, 12))
        cb.bind("<<ComboboxSelected>>", self._apply_preset)

        fields = [
            ("alarm-s:", self.var_alarm_s),
            ("window-s:", self.var_window_s),
            ("th-fall:", self.var_th_fall),
            ("th-down:", self.var_th_down),
            ("n-fall:", self.var_n_fall),
            ("n-down:", self.var_n_down),
            ("smooth:", self.var_smooth),
            ("v-gate:", self.var_v_gate),
        ]
        for lbl, var in fields:
            tk.Label(r2, text=lbl, bg="#F1F5F9", font=("맑은 고딕", 9)).pack(side=tk.LEFT)
            ttk.Entry(r2, textvariable=var, width=5).pack(side=tk.LEFT, padx=(2, 8))

        r3 = tk.Frame(inner, bg="#F1F5F9")
        r3.pack(fill=tk.X, pady=(4, 0))
        tk.Label(r3, text="동작 옵션:", bg="#F1F5F9", font=("맑은 고딕", 9, "bold"), width=11, anchor=tk.W).pack(side=tk.LEFT)
        opts = [
            ("속도 조건(v-gate)", self.var_use_v_gate),
            ("ID 이어붙이기", self.var_stitch),
            ("1인 영상 모드", self.var_single_person),
            ("경보 유지(--latch-alarm)", self.var_latch_alarm),
            ("결과 MP4/CSV 저장", self.var_save_output),
            ("경보 알림음", self.var_sound_alarm),
        ]
        for txt, var in opts:
            tk.Checkbutton(r3, text=txt, variable=var, bg="#F1F5F9", font=("맑은 고딕", 9)).pack(side=tk.LEFT, padx=(0, 12))

    # -------------------- 로딩 창(오버레이) 애니메이션 --------------------
    def _start_loading_overlay(self, subtext="잠시만 기다려 주세요..."):
        self.is_loading = True
        self.loading_angle = 0
        self.loading_subtext = subtext
        self._animate_loading_overlay()

    def _update_loading_subtext(self, subtext):
        self.loading_subtext = subtext

    def _stop_loading_overlay(self):
        self.is_loading = False
        if self.loading_after_id is not None:
            try:
                self.root.after_cancel(self.loading_after_id)
            except Exception:
                pass
            self.loading_after_id = None

    def _animate_loading_overlay(self):
        if not self.is_loading:
            return

        c = self.video_canvas
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw > 20 and ch > 20:
            c.delete("all")
            # 좌측 상단 기본 라벨 유지
            c.create_text(
                18,
                16,
                text="영상 + 관절 오버레이",
                fill="#CBD5E1",
                font=("맑은 고딕", 11, "bold"),
                anchor=tk.NW,
            )

            cx, cy = cw // 2, ch // 2
            box_w, box_h = 340, 175
            x0, y0 = cx - box_w // 2, cy - box_h // 2
            x1, y1 = cx + box_w // 2, cy + box_h // 2

            # 중앙 로딩 팝업 카드 박스
            c.create_rectangle(x0, y0, x1, y1, fill="#181D24", outline="#3B82F6", width=2)

            # 회전 스피너 원형 트랙 + 회전 아크
            r = 22
            sc_y = y0 + 52
            c.create_oval(cx - r, sc_y - r, cx + r, sc_y + r, outline="#334155", width=4)
            c.create_arc(
                cx - r,
                sc_y - r,
                cx + r,
                sc_y + r,
                start=self.loading_angle,
                extent=100,
                style=tk.ARC,
                outline="#3B82F6",
                width=4,
            )

            # 로딩 안내 문구
            c.create_text(
                cx,
                y0 + 105,
                text="모델 로딩 중...",
                fill="#FFFFFF",
                font=("맑은 고딕", 13, "bold"),
            )
            c.create_text(
                cx,
                y0 + 135,
                text=self.loading_subtext,
                fill="#94A3B8",
                font=("맑은 고딕", 9),
            )

            self.loading_angle = (self.loading_angle - 18) % 360

        self.loading_after_id = self.root.after(40, self._animate_loading_overlay)

    # -------------------- UI 이벤트 핸들러 --------------------
    def _toggle_settings_drawer(self):
        self.show_settings = not self.show_settings
        if self.show_settings:
            self.settings_drawer.pack(fill=tk.X, side=tk.TOP, after=self.root.winfo_children()[1])
            self.btn_toggle_cfg.config(text="상세 설정 ▴")
        else:
            self.settings_drawer.pack_forget()
            self.btn_toggle_cfg.config(text="상세 설정 ▾")

    def _toggle_source(self):
        if self.var_source_type.get() == "video":
            self.ent_video.config(state="normal")
            self.btn_browse_vid.config(state="normal")
            self.spn_cam.config(state="disabled")
        else:
            self.ent_video.config(state="disabled")
            self.btn_browse_vid.config(state="disabled")
            self.spn_cam.config(state="normal")

    def _browse_video(self):
        init_dir = str((ROOT.parent / "3_테스트영상").resolve()) if (ROOT.parent / "3_테스트영상").exists() else str(ROOT)
        path = filedialog.askopenfilename(
            initialdir=init_dir,
            title="테스트 영상 선택",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")],
        )
        if path:
            self.var_video_path.set(path)

    def _browse_models(self):
        paths = filedialog.askopenfilenames(title="TCNTE 모델(best.pt) 선택", filetypes=[("PyTorch Model", "*.pt")])
        if paths:
            self.var_models.set(" ; ".join(paths))

    def _browse_weights(self):
        path = filedialog.askopenfilename(title="YOLO Pose 가중치 선택", filetypes=[("PyTorch Model", "*.pt")])
        if path:
            self.var_weights.set(path)

    def _apply_preset(self, _event=None):
        p = self.var_preset.get()
        if p == "final":
            self.var_th_fall.set(0.5)
            self.var_window_s.set(5.0)
            self.var_use_v_gate.set(True)
            self.var_v_gate.set(0.02)
        elif p == "B":
            self.var_th_fall.set(0.4)
            self.var_window_s.set(4.0)
            self.var_use_v_gate.set(True)
            self.var_v_gate.set(0.02)
        elif p == "v1":
            self.var_th_fall.set(0.5)
            self.var_window_s.set(3.0)
            self.var_use_v_gate.set(False)

    def _on_video_canvas_resize(self, event):
        if not self.is_running and not self.is_loading and self.current_photo is None:
            self.video_canvas.coords(self.center_hint_id, event.width // 2, event.height // 2)

    # -------------------- 카드 1, 2, 3 및 게이지 렌더링 --------------------
    def _update_alert_card(self, state, tid=None, elapsed_down_s=0.0):
        cam_name = self.var_camera_name.get().strip() or "카메라"
        now_str = time.strftime("%H:%M:%S")
        id_txt = f"ID {tid}" if tid is not None else "대상 없음"

        if state == "ALARM":
            bg = "#C2410C"
            title = f"낙상 경보 · {id_txt}"
            sub = f"{now_str} · {cam_name}"
        elif state == "DOWN":
            bg = "#EA580C"
            title = f"쓰러짐 확인 · {id_txt}"
            sub = f"지속 {elapsed_down_s:.1f}초 · {cam_name}"
        elif state == "SUSPECT":
            bg = "#D97706"
            title = f"낙상 의심 · {id_txt}"
            sub = f"{now_str} · {cam_name}"
        elif state == "NORMAL":
            bg = "#15803D"
            title = f"정상 감시 중 · {id_txt}"
            sub = f"{now_str} · {cam_name}"
        elif state == "LOADING":
            bg = "#1E293B"
            title = "모델 로딩 중..."
            sub = f"{now_str} · {cam_name}"
        else:
            bg = "#1E293B"
            title = "감지 대상 없음"
            sub = f"{now_str} · {cam_name}"

        def _apply():
            self.alert_card.config(bg=bg)
            self.lbl_alert_title.config(text=title, bg=bg)
            self.lbl_alert_sub.config(text=sub, bg=bg)

        self.root.after(0, _apply)

    def _draw_prob_graph(self):
        c = self.graph_canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 30 or h < 30:
            return

        c.delete("all")
        pad_l, pad_r, pad_t, pad_b = 14, 14, 14, 18
        gw = max(1, w - pad_l - pad_r)
        gh = max(1, h - pad_t - pad_b)

        th = float(self.var_th_down.get())
        y_th = pad_t + gh * (1.0 - min(max(th, 0.0), 1.0))

        c.create_line(pad_l, y_th, w - pad_r, y_th, fill="#CBD5E1", dash=(4, 4), width=1.5)
        c.create_text(
            w - pad_r - 2,
            y_th + 12,
            text=f"임계 {th:.1f}",
            fill="#64748B",
            font=("맑은 고딕", 8, "bold"),
            anchor=tk.E,
        )

        pts = list(self.prob_history)
        if len(pts) >= 2:
            max_len = self.prob_history.maxlen
            coords = []
            start_idx = max_len - len(pts)
            for idx, val in enumerate(pts):
                if np.isnan(val):
                    continue
                x = pad_l + gw * ((start_idx + idx) / max(1, max_len - 1))
                y = pad_t + gh * (1.0 - min(max(float(val), 0.0), 1.0))
                coords.extend([x, y])

            if len(coords) >= 4:
                line_col = "#C2410C" if (pts[-1] >= th) else "#2563EB"
                c.create_line(*coords, fill=line_col, width=2.8, smooth=True)

    def append_event_log(self, text):
        def _ins():
            self.txt_log.config(state="normal")
            self.txt_log.insert(tk.END, text + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state="disabled")

        self.root.after(0, _ins)

    def _render_frame_to_canvas(self, bgr_img):
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        h, w = bgr_img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        def _draw():
            if self.is_loading:
                self._stop_loading_overlay()
            self.current_photo = img_tk
            self.video_canvas.delete("all")
            self.video_canvas.create_image(cw // 2, ch // 2, image=img_tk, anchor=tk.CENTER)
            self.video_canvas.create_text(
                18,
                16,
                text="영상 + 관절 오버레이",
                fill="#E2E8F0",
                font=("맑은 고딕", 11, "bold"),
                anchor=tk.NW,
            )

        self.root.after(0, _draw)

    def _play_beep(self):
        if not self.var_sound_alarm.get():
            return
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.MessageBeep(winsound.MB_ICONHAND)
            else:
                self.root.bell()
        except Exception:
            pass

    # -------------------- 실행 제어 --------------------
    def _build_runtime_cfg(self):
        class Cfg:
            pass

        c = Cfg()
        c.smooth = int(self.var_smooth.get())
        c.th_fall = float(self.var_th_fall.get())
        c.th_down = float(self.var_th_down.get())
        c.n_fall = int(self.var_n_fall.get())
        c.n_down = int(self.var_n_down.get())
        c.window_s = float(self.var_window_s.get())
        c.v_gate = float(self.var_v_gate.get()) if self.var_use_v_gate.get() else None
        c.alarm_s = float(self.var_alarm_s.get())
        c.stitch = bool(self.var_stitch.get())
        c.alias_iou = 0.5
        c.single_person = bool(self.var_single_person.get())
        c.fix_release = bool(self.var_fix_release.get())
        c.latch_alarm = bool(self.var_latch_alarm.get())
        return c

    def start_detection(self):
        if self.is_running:
            return

        if self.var_source_type.get() == "video":
            vpath = self.var_video_path.get().strip()
            if not vpath or not Path(vpath).exists():
                messagebox.showerror("오류", "유효한 동영상 파일을 선택해 주세요.")
                return

        model_paths = [m.strip() for m in self.var_models.get().split(";") if m.strip()]
        if not model_paths or not all(Path(m).exists() for m in model_paths):
            messagebox.showerror("오류", "TCNTE 모델 파일(.pt) 경로를 확인해 주세요.")
            return

        wpath = self.var_weights.get().strip()
        if not wpath:
            messagebox.showerror("오류", "YOLO Pose 가중치 경로를 확인해 주세요.")
            return

        self.is_running = True
        self.is_paused = False
        self.current_photo = None
        self.prob_history.clear()
        self.prev_person_states.clear()
        self.alarm_triggered_ids.clear()
        self.var_progress.set(0.0)

        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal", text="⏸ 일시정지")
        self.btn_stop.config(state="normal")

        # 영상 영역 로딩 창 즉시 표시
        self._update_alert_card("LOADING")
        self._start_loading_overlay("YOLO26n-pose 및 TCNTE 모델 준비 중...")

        cfg = self._build_runtime_cfg()
        self.worker_thread = threading.Thread(
            target=self._inference_loop, args=(cfg, model_paths, wpath), daemon=True
        )
        self.worker_thread.start()

    def toggle_pause(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        self.btn_pause.config(text="▶ 재개" if self.is_paused else "⏸ 일시정지")

    def stop_detection(self):
        if self.is_running:
            self.is_running = False
            self.is_paused = False

    def _reset_ui_on_finish(self):
        def _ui():
            self._stop_loading_overlay()
            self.btn_start.config(state="normal")
            self.btn_pause.config(state="disabled", text="⏸ 일시정지")
            self.btn_stop.config(state="disabled")

        self.root.after(0, _ui)

    # -------------------- 핵심 추론 루프 --------------------
    def _inference_loop(self, a, model_paths, wpath):
        vw = None
        fcsv = None
        cap = None
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.root.after(0, lambda: self.var_progress_text.set("모델 로딩 중..."))

            self.root.after(0, lambda: self._update_loading_subtext(f"YOLO26n-pose 모델 로딩 중... ({dev.upper()})"))
            yolo = YOLO(wpath if Path(wpath).exists() else "yolo26n-pose.pt")

            self.root.after(0, lambda: self._update_loading_subtext(f"TCNTE 앙상블 모델 ({len(model_paths)}개) 로딩 중..."))
            models = load_models(model_paths, dev)

            self.root.after(0, lambda: self._update_loading_subtext("영상 스트림 및 첫 프레임 준비 중..."))

            src_type = self.var_source_type.get()
            total_target_frames = 0

            if src_type == "video":
                vpath = Path(self.var_video_path.get().strip())
                name = vpath.stem
                input_label = vpath.name

                _tmp_cap = cv2.VideoCapture(str(vpath))
                src_fc = _tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                src_fps = _tmp_cap.get(cv2.CAP_PROP_FPS) or 30.0
                _tmp_cap.release()
                if src_fc > 0 and src_fps > 0:
                    total_target_frames = max(1, int(round((src_fc / src_fps) * FPS)))

                self.root.after(0, lambda: self.progressbar.config(mode="determinate", maximum=100.0))
                try:
                    frames = (img for _, img, _ in extract_mod.frames_pyav(vpath))
                except Exception:
                    frames = (img for _, img, _ in extract_mod.frames_cv2(vpath))
            else:
                cam_idx = int(self.var_webcam_idx.get())
                name = f"webcam{cam_idx}"
                input_label = f"웹캠 {cam_idx}"
                cap = cv2.VideoCapture(cam_idx)
                if not cap.isOpened():
                    raise RuntimeError(f"웹캠({cam_idx})을 열 수 없습니다.")

                self.root.after(0, lambda: self.progressbar.config(mode="indeterminate"))
                self.root.after(0, lambda: self.progressbar.start(15))

                def cam_gen():
                    t0, k = time.time(), 0
                    while self.is_running:
                        ok, img = cap.read()
                        if not ok:
                            break
                        if time.time() - t0 >= k / FPS:
                            k += 1
                            yield img

                frames = cam_gen()

            out_dir = LOGS / "demo"
            if self.var_save_output.get():
                out_dir.mkdir(parents=True, exist_ok=True)
                fcsv = open(out_dir / f"{name}_frames.csv", "w", newline="", encoding="utf-8")
                cw = csv.writer(fcsv)
                cw.writerow(
                    ["frame", "t", "id", "p_normal", "p_fall", "p_fallen", "state"]
                    + (["vmax"] if a.v_gate is not None else [])
                )

            alias = {}
            n_link = {"gap": 0, "iou": 0, "single": 0}
            people = {}
            n_alarm = set()
            t_last = time.time()

            for i, img in enumerate(frames):
                if not self.is_running:
                    break
                while self.is_paused and self.is_running:
                    time.sleep(0.05)

                t_step0 = time.time()
                r = yolo.track(
                    img, persist=(i > 0), tracker="bytetrack.yaml", verbose=False, device=0 if dev == "cuda" else "cpu"
                )[0]
                ids = []

                if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                    kxy = r.keypoints.xy.cpu().numpy()
                    kc = r.keypoints.conf.cpu().numpy()
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    raw_ids = [int(t) for t in r.boxes.id.int().cpu().numpy()]
                    seen_now = {alias.get(t, t) for t in raw_ids}
                    order = list(range(len(raw_ids)))

                    if a.stitch and a.alias_iou > 0:
                        order.sort(
                            key=lambda j: 0
                            if (raw_ids[j] in people and raw_ids[j] not in alias)
                            else (1 if raw_ids[j] in alias else 2)
                        )

                    for j in order:
                        tid = raw_ids[j]
                        if a.stitch:
                            if tid in alias:
                                tid = alias[tid]
                            elif tid not in people:
                                if a.alias_iou > 0:
                                    hit = next(
                                        (k for k in ids if box_iou(people[k].frames[i][2], xyxy[j]) >= a.alias_iou),
                                        None,
                                    )
                                    if hit is not None:
                                        alias[tid] = hit
                                        n_link["iou"] += 1
                                        continue
                                b = xyxy[j]
                                cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                                best = None
                                for k, p in people.items():
                                    if k in ids or k in seen_now or not (0 < i - p.last_seen <= MERGE_GAP):
                                        continue
                                    la = p.frames[p.last_seen][2]
                                    ca = np.array([(la[0] + la[2]) / 2, (la[1] + la[3]) / 2])
                                    if np.linalg.norm(ca - cb) < MERGE_DIST * max(la[3] - la[1], 1.0) and (
                                        best is None or p.last_seen > best[0]
                                    ):
                                        best = (p.last_seen, k)
                                if best:
                                    alias[tid] = best[1]
                                    tid = best[1]
                                    n_link["gap"] += 1
                                elif a.single_person and not (seen_now & set(people)):
                                    last = max(
                                        ((p.last_seen, k) for k, p in people.items() if k not in ids), default=None
                                    )
                                    if last is not None:
                                        alias[tid] = last[1]
                                        tid = last[1]
                                        n_link["single"] += 1
                            if tid in ids:
                                continue

                        people.setdefault(tid, Person(a)).add(i, kxy[j], kc[j], xyxy[j])
                        ids.append(tid)

                if a.stitch and a.single_person and ids:
                    done = set()
                    for k in [k for k, p in people.items() if k not in ids and 0 < i - p.last_seen <= MERGE_GAP]:
                        pk = people[k]
                        cand = [
                            j
                            for j in ids
                            if j not in done
                            and min(people[j].frames) > min(pk.frames)
                            and len(set(people[j].frames) & set(pk.frames)) <= MERGE_GAP
                        ]
                        if not cand:
                            continue
                        la = pk.frames[pk.last_seen][2]
                        j = min(
                            cand,
                            key=lambda j: np.hypot(
                                (people[j].frames[i][2][0] + people[j].frames[i][2][2] - la[0] - la[2]) / 2,
                                (people[j].frames[i][2][1] + people[j].frames[i][2][3] - la[1] - la[3]) / 2,
                            ),
                        )
                        pk.add(i, *people[j].frames[i])
                        for raw, key in list(alias.items()):
                            if key == j:
                                alias[raw] = k
                        alias[j] = k
                        del people[j]
                        ids[ids.index(j)] = k
                        n_link["single"] += 1
                        done.add(k)

                for tid in ids:
                    pr = people[tid]
                    prev_st = self.prev_person_states.get(tid, "NORMAL")
                    X = compute_features(*pr.arrays(i))
                    vmax = float(X[-16:, 54].max())
                    st = pr.step(i, predict(models, X, dev), X[-1, 51], vmax)
                    self.prev_person_states[tid] = st

                    now_ts = time.strftime("%H:%M:%S")
                    if st == "SUSPECT" and prev_st == "NORMAL":
                        self.append_event_log(f"{now_ts} ID {tid} 낙상 의심 감지")
                    elif st == "NORMAL" and prev_st in ("SUSPECT", "DOWN", "ALARM"):
                        self.append_event_log(f"{now_ts} ID {tid} {prev_st} → 해제")
                    elif st == "ALARM":
                        n_alarm.add(tid)
                        if prev_st != "ALARM":
                            delay_s = (i - pr.sus_t) / FPS if pr.sus_t is not None else a.alarm_s
                            self.append_event_log(f"{now_ts} ID {tid} 낙상 경보 (지연 {delay_s:.1f}초)")
                            self._play_beep()

                    if fcsv is not None:
                        cw.writerow(
                            [i, round(i / FPS, 3), tid, *np.round(pr.p, 4), st]
                            + ([round(vmax, 4)] if a.v_gate is not None else [])
                        )

                for tid in [t for t, p in people.items() if i - p.last_seen > LOST_DROP]:
                    del people[tid]
                    self.prev_person_states.pop(tid, None)
                if alias:
                    alias = {k: v for k, v in alias.items() if v in people}

                now = time.time()
                infer_ms = (now - t_step0) * 1000.0
                cur_fps = 1.0 / max(now - t_last, 1e-6)
                t_last = now

                vis = draw(img.copy(), people, ids, cv2, i, f"{cur_fps:4.1f} fps")
                if self.var_save_output.get():
                    if vw is None:
                        h, w = vis.shape[:2]
                        vw = cv2.VideoWriter(
                            str(out_dir / f"{name}_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h)
                        )
                    vw.write(vis)

                if ids:
                    focus_id = max(ids, key=lambda t: (SEVERITY.index(people[t].state), len(people[t].frames)))
                    fpr = people[focus_id]
                    fall_prob = float(min(1.0, fpr.p[1] + fpr.p[2]))
                    self.prob_history.append(fall_prob)
                    down_dur = (i - fpr.down_t) / FPS if (fpr.down_t is not None and fpr.state in ("DOWN", "ALARM")) else 0.0
                    self._update_alert_card(fpr.state, focus_id, down_dur)
                    self.root.after(0, lambda fid=focus_id: self.lbl_graph_title.config(text=f"ID {fid} 낙상 확률 (최근 10초)"))
                else:
                    self.prob_history.append(np.nan)
                    self._update_alert_card("NONE", None, 0.0)
                    self.root.after(0, lambda: self.lbl_graph_title.config(text="ID - 낙상 확률 (최근 10초)"))

                if src_type == "video":
                    if total_target_frames > 0:
                        pct = min(100.0, ((i + 1) / total_target_frames) * 100.0)
                        prog_str = f"{pct:5.1f}%  ({i + 1} / {total_target_frames} 프레임 | {i / FPS:.1f}s / {total_target_frames / FPS:.1f}s)"
                    else:
                        pct = 0.0
                        prog_str = f"분석 중... ({i + 1} 프레임 | {i / FPS:.1f}s)"
                    self.root.after(0, lambda v=pct, s=prog_str: (self.var_progress.set(v), self.var_progress_text.set(s)))
                else:
                    prog_str = f"실시간 스트리밍 중 ({i + 1} 프레임 | 경과 {i / FPS:.1f}s)"
                    self.root.after(0, lambda s=prog_str: self.var_progress_text.set(s))

                hdr_txt = f"입력: {input_label} | {cur_fps:.1f} FPS | {infer_ms:.0f} ms"
                self.root.after(0, lambda t=hdr_txt: self.lbl_header_status.config(text=t))
                self.root.after(0, self._draw_prob_graph)
                self._render_frame_to_canvas(vis)

            if src_type == "video" and self.is_running:
                self.root.after(0, lambda: (self.var_progress.set(100.0), self.var_progress_text.set("100.0%  (분석 완료)")))

            if self.var_save_output.get() and a.stitch:
                with open(out_dir / f"{name}_info.json", "w", encoding="utf-8") as jf:
                    json.dump(
                        dict(
                            stitch_gap=n_link["gap"],
                            stitch_iou=n_link["iou"],
                            stitch_single=n_link["single"],
                            alarm_ids=sorted(n_alarm),
                        ),
                        jf,
                        indent=1,
                    )

            self.append_event_log(f"{time.strftime('%H:%M:%S')} 분석 종료 (경보 ID: {sorted(n_alarm) or '없음'})")

        except Exception as e:
            self.append_event_log(f"{time.strftime('%H:%M:%S')} [오류] {e}")
            self.root.after(0, lambda err=str(e): messagebox.showerror("실행 오류", err))
        finally:
            if cap is not None:
                cap.release()
            if vw is not None:
                vw.release()
            if fcsv is not None:
                fcsv.close()
            if self.var_source_type.get() == "webcam":
                self.root.after(0, self.progressbar.stop)
            self.is_running = False
            self._reset_ui_on_finish()

    def on_close(self):
        self.is_running = False
        self._stop_loading_overlay()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FallMonitorGUI(root)
    root.mainloop()