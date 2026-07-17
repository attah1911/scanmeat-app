"""
ScanMeat!
Klasifikasi Tingkat Kesegaran Daging Sapi Berbasis Citra Menggunakan YOLOv8s
──────────────────────────────────────────────────────────────────────────────
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import cv2
import io
from pathlib import Path
from PIL import Image

# ─── Konfigurasi Halaman ────────────────────────────────────────────────────

st.set_page_config(
    page_title="ScanMeat! — Deteksi Kesegaran Daging Sapi",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Konstanta ───────────────────────────────────────────────────────────────

# MODEL_PATH = Path("models/part9(fix)/best.pt")
MODEL_PATH = Path("models/part9(fix)/best.pt")
ERROR_THRESHOLD = 0.35
WARNING_THRESHOLD = 0.50

CLASS_COLORS = {
    "Segar": (34, 197, 94),        # hijau (RGB)
    "Setengah Segar": (234, 179, 8),  # kuning (RGB)
    "Busuk": (239, 68, 68),        # merah (RGB)
}

CLASS_ICONS = {
    "Segar": "●",
    "Setengah Segar": "●",
    "Busuk": "●",
}

# ─── Custom CSS ──────────────────────────────────────────────────────────────

def inject_custom_css():
    """Menyuntikkan custom CSS untuk tampilan profesional bertema merah/daging."""
    st.markdown("""
    <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Root Variables ── */
        :root {
            --primary: #1e0a0a; /* Dark reddish slate */
            --primary-light: #3b1818;
            --accent: #dc2626; /* Bright crimson red */
            --accent-hover: #b91c1c;
            --surface: #fdfbfb;
            --surface-card: #ffffff;
            --text-primary: #1e0a0a;
            --text-secondary: #5c4d4d;
            --border: #f1e5e5;
            --success: #16a34a;
            --warning: #ca8a04;
            --danger: #dc2626;
            --radius: 12px;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.03), 0 1px 2px rgba(0,0,0,0.02);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.04), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        }

        /* ── Global Styles ── */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--surface);
        }

        /* ── Sembunyikan Sidebar & Tombol Toggle ── */
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }

        /* ── Header App ── */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem 0 1rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 2rem;
        }

        .brand-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--primary);
            letter-spacing: -0.03em;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .brand-title span {
            color: var(--accent);
        }

        .brand-subtitle {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 500;
            background: #fef2f2;
            color: var(--accent);
            padding: 4px 12px;
            border-radius: 20px;
        }

        /* ── Hero Banner ── */
        .hero-banner {
            background: linear-gradient(135deg, #1e0a0a 0%, #4c1d1d 50%, #2d1010 100%);
            padding: 2.5rem 2rem;
            border-radius: var(--radius);
            margin-bottom: 2rem;
            color: white;
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-md);
        }

        .hero-banner h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.02em;
        }

        .hero-banner p {
            font-size: 0.95rem;
            color: #fca5a5;
            margin: 0;
            max-width: 700px;
            line-height: 1.5;
        }

        /* ── Custom Tabs / Navbar ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: 45px;
            padding: 0 24px;
            background-color: transparent;
            border-radius: 8px 8px 0 0;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-secondary);
            transition: all 0.2s ease;
            border: none;
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 2px solid var(--accent) !important;
        }

        /* ── Card Component ── */
        .info-card {
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-sm);
        }

        .info-card h3 {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 0;
            margin-bottom: 1rem;
            border-left: 4px solid var(--accent);
            padding-left: 8px;
        }

        .info-card p {
            font-size: 0.9rem;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        /* ── Status Bounding Box / Results ── */
        .result-container {
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
        }

        .label-tag {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 700;
        }

        .tag-segar {
            background: #dcfce7;
            color: #16a34a;
        }

        .tag-setengah {
            background: #fef9c3;
            color: #ca8a04;
        }

        .tag-busuk {
            background: #fee2e2;
            color: #dc2626;
        }

        .dot-icon {
            font-size: 0.7rem;
            vertical-align: middle;
        }

        .detection-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.85rem 1.25rem;
            background: #fdfcfc;
            border-radius: 8px;
            margin-bottom: 0.75rem;
            border: 1px solid var(--border);
        }

        .detection-item .det-label {
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--text-primary);
        }

        .detection-item .det-conf {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        /* ── Progress Bar ── */
        .conf-bar-container {
            width: 100px;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            display: inline-block;
        }

        .conf-bar {
            height: 100%;
            border-radius: 4px;
        }

        .conf-bar-success { background: var(--success); }
        .conf-bar-warning { background: var(--warning); }
        .conf-bar-danger  { background: var(--danger); }

        /* ── Footer ── */
        .app-footer {
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.8rem;
            font-weight: 500;
        }

        /* ── Grid/Table Info ── */
        .threshold-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        .threshold-table th, .threshold-table td {
            text-align: left;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
        }

        .threshold-table th {
            font-weight: 600;
            color: var(--text-primary);
            background: #fdfcfc;
        }

        .threshold-table td {
            color: var(--text-secondary);
        }

        .class-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* ── Streamlit Overrides ── */
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1.25rem;
        }
    </style>
    """, unsafe_allow_html=True)


# ─── Fungsi Utilitas ─────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Memuat model YOLOv8 dari path yang ditentukan dengan caching resource."""
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        return None, f"File model tidak ditemukan di: {MODEL_PATH.resolve()}"

    try:
        model = YOLO(str(MODEL_PATH))
        return model, None
    except Exception as e:
        return None, f"Gagal memuat model: {str(e)}"


def process_image(image_input) -> np.ndarray:
    """Mengubah input file/kamera menjadi format NumPy array (BGR)."""
    img_bytes = image_input.getvalue()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img_bgr


def draw_boxes_and_filter(img_bgr: np.ndarray, results) -> tuple:
    """
    Menggambar bounding box hasil deteksi dan memfilter berdasarkan threshold confidence.
    
    Returns:
        tuple: (annotated_image_rgb, detections_list, max_confidence, status)
    """
    img_annotated = img_bgr.copy()
    detections = []

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                detections.append({"class": cls_name, "confidence": conf})

    # Tentukan confidence tertinggi
    if not detections:
        max_conf = 0.0
    else:
        max_conf = max(d["confidence"] for d in detections)

    # Tentukan status berdasarkan dual threshold
    if max_conf < ERROR_THRESHOLD:
        status = "error"
    elif max_conf < WARNING_THRESHOLD:
        status = "warning"
    else:
        status = "success"

    # Gambar bounding box hanya jika di atas ERROR_THRESHOLD
    if status != "error" and detections and results:
        result = results[0]
        boxes = result.boxes

        # Hitung parameter adaptif berdasarkan resolusi gambar
        img_h, img_w = img_annotated.shape[:2]
        img_diag = (img_h ** 2 + img_w ** 2) ** 0.5
        box_thickness = max(3, int(img_diag / 400))
        font_scale = max(0.8, img_diag / 1800)
        text_thickness = max(2, int(img_diag / 600))
        label_pad = max(8, int(img_diag / 200))

        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, f"class_{cls_id}")

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Warna BGR
            color = CLASS_COLORS.get(cls_name, (128, 128, 128))
            color_bgr = (color[2], color[1], color[0])

            # Gambar box
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color_bgr, box_thickness)

            # Label teks
            label = f"{cls_name} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

            # Background label
            cv2.rectangle(
                img_annotated,
                (x1, y1 - th - label_pad * 2),
                (x1 + tw + label_pad, y1),
                color_bgr,
                -1,
            )
            # Teks label putih
            cv2.putText(
                img_annotated,
                label,
                (x1 + label_pad // 2, y1 - label_pad // 2),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
            )

    # Konversi ke RGB untuk Streamlit
    img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
    return img_rgb, detections, max_conf, status


def get_conf_bar_class(conf: float) -> str:
    """Mengembalikan class CSS untuk confidence bar."""
    if conf >= WARNING_THRESHOLD:
        return "conf-bar-success"
    elif conf >= ERROR_THRESHOLD:
        return "conf-bar-warning"
    return "conf-bar-danger"


def get_tag_class(cls_name: str) -> str:
    """Mengembalikan class CSS untuk warna tag hasil."""
    mapping = {
        "Segar": "tag-segar",
        "Setengah Segar": "tag-setengah",
        "Busuk": "tag-busuk",
    }
    return mapping.get(cls_name, "")


# ─── Komponen Halaman ────────────────────────────────────────────────────────

def render_top_navbar():
    """Merender Header Brand Aplikasi."""
    st.markdown("""
    <div class="app-header">
        <div class="brand-title">Scan<span>Meat!</span></div>
        <div class="brand-subtitle">YOLOv8s Object Detection</div>
    </div>
    """, unsafe_allow_html=True)


def render_beef_characteristics():
    """Merender perbandingan ciri fisik daging sapi segar, setengah segar, dan busuk dengan gambar referensi."""
    st.markdown("---")
    st.markdown("""
    <h3 style='font-size: 1.2rem; font-weight: 700; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 8px;'>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>
        </svg>
        Panduan Ciri Fisik Daging Sapi
    </h3>
    <p style='font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1.25rem;'>Bandingkan hasil deteksi model dengan ciri-ciri fisik daging berikut sebagai verifikasi manual.</p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("assets/segar.jpg", caption="Contoh Daging Sapi Segar", use_container_width=True)
        st.markdown("""
        <div class="info-card" style="border-top: 4px solid #16a34a;">
            <h4 style="color: #16a34a; font-weight: 700; margin-top:0; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                Daging Segar
            </h4>
            <ul style="font-size: 0.85rem; color: var(--text-secondary); padding-left: 1.1rem; line-height: 1.7; margin: 0 0 0.75rem 0;">
                <li><strong>Warna:</strong> Merah cerah/terang, mengkilap dan merata.</li>
                <li><strong>Tekstur:</strong> Kenyal, padat, dan sangat elastis (kembali ke bentuk semula jika ditekan).</li>
                <li><strong>Aroma:</strong> Khas aroma daging sapi segar, tidak berbau amis tajam, asam, atau busuk.</li>
                <li><strong>Permukaan:</strong> Bersih, tidak berlendir, tidak berair berlebih.</li>
            </ul>
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 8px 12px; font-size: 0.8rem; color: #166534; display: flex; align-items: flex-start; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2" style="flex-shrink:0; margin-top: 2px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                <span><strong>Aman dikonsumsi.</strong> Daging dalam kondisi segar layak untuk dimasak dan dikonsumsi secara langsung.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image("assets/setengah_segar.jpg", caption="Contoh Daging Sapi Setengah Segar", use_container_width=True)
        st.markdown("""
        <div class="info-card" style="border-top: 4px solid #ca8a04;">
            <h4 style="color: #ca8a04; font-weight: 700; margin-top:0; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ca8a04" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                Daging Setengah Segar
            </h4>
            <ul style="font-size: 0.85rem; color: var(--text-secondary); padding-left: 1.1rem; line-height: 1.7; margin: 0 0 0.75rem 0;">
                <li><strong>Warna:</strong> Mulai pucat atau sedikit kecokelatan/kusam di beberapa area.</li>
                <li><strong>Tekstur:</strong> Agak lunak, elastisitas mulai menurun (lambat kembali setelah ditekan).</li>
                <li><strong>Aroma:</strong> Mulai tercium aroma asam atau bau segar khasnya sudah berkurang.</li>
                <li><strong>Permukaan:</strong> Terasa agak lembap atau sedikit basah namun belum berlendir lengket.</li>
            </ul>
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 8px 12px; font-size: 0.8rem; color: #991b1b; display: flex; align-items: flex-start; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#991b1b" stroke-width="2" style="flex-shrink:0; margin-top: 2px;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
                <span><strong>Tidak layak dikonsumsi.</strong> Daging setengah segar menunjukkan penurunan kualitas yang signifikan. Jangan dikonsumsi untuk menghindari risiko kesehatan.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.image("assets/busuk.jpg", caption="Contoh Daging Sapi Busuk", use_container_width=True)
        st.markdown("""
        <div class="info-card" style="border-top: 4px solid #dc2626;">
            <h4 style="color: #dc2626; font-weight: 700; margin-top:0; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
                Daging Busuk
            </h4>
            <ul style="font-size: 0.85rem; color: var(--text-secondary); padding-left: 1.1rem; line-height: 1.7; margin: 0 0 0.75rem 0;">
                <li><strong>Warna:</strong> Cokelat gelap keabu-abuan, atau muncul noda kehijauan/hitam.</li>
                <li><strong>Tekstur:</strong> Sangat lembek, berlendir tebal, lengket, dan tidak elastis sama sekali.</li>
                <li><strong>Aroma:</strong> Bau busuk menyengat, asam pekat, atau berbau seperti amonia/bangkai.</li>
                <li><strong>Permukaan:</strong> Berlendir pekat dan terasa lengket saat disentuh.</li>
            </ul>
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 8px 12px; font-size: 0.8rem; color: #991b1b; display: flex; align-items: flex-start; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#991b1b" stroke-width="2" style="flex-shrink:0; margin-top: 2px;"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
                <span><strong>Tidak layak dikonsumsi.</strong> Daging busuk mengandung bakteri patogen berbahaya. Buang dan jangan dikonsumsi dalam kondisi apapun.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_scan_tab(model):
    """Merender konten untuk tab Scan Daging."""
    # Banner Hero
    st.markdown("""
    <div class="hero-banner">
        <h1>Klasifikasi Tingkat Kesegaran Daging Sapi</h1>
        <p>Gunakan kamera perangkat Anda atau unggah berkas foto daging sapi untuk mendeteksi tingkat kesegaran secara instan.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pilihan Mode Input
    col_opt, _ = st.columns([2, 3])
    with col_opt:
        input_mode = st.radio(
            "Metode Input Gambar",
            ["Unggah Foto", "Kamera Perangkat"],
            horizontal=True,
            label_visibility="visible",
        )

    st.markdown("")

    image_input = None

    if input_mode == "Unggah Foto":
        image_input = st.file_uploader(
            "Pilih file gambar daging sapi",
            type=["jpg", "jpeg", "png"],
            key="file_uploader",
            help="Mendukung format JPG, JPEG, atau PNG.",
        )
    else:
        # ── Inisialisasi session state untuk facingMode ──
        if "cam_facing" not in st.session_state:
            st.session_state.cam_facing = "environment"

        # ── Tombol balik kamera ──
        facing_label = "🔄 Kamera Depan" if st.session_state.cam_facing == "environment" else "🔄 Kamera Belakang"
        if st.button(facing_label, key="btn_switch_cam"):
            st.session_state.cam_facing = "user" if st.session_state.cam_facing == "environment" else "environment"
            st.rerun()

        facing_info = "Kamera Belakang" if st.session_state.cam_facing == "environment" else "Kamera Depan"
        st.caption(f"Kamera aktif: **{facing_info}** — Klik tombol di atas untuk beralih.")

        # ── Inject JS untuk intercept getUserMedia sebelum st.camera_input init ──
        facing_val = st.session_state.cam_facing
        st.markdown(f"""
        <script>
        (function(){{
            var _facing = '{facing_val}';
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {{
                var _orig = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                navigator.mediaDevices.getUserMedia = function(constraints) {{
                    if (constraints && constraints.video) {{
                        if (typeof constraints.video === 'boolean') {{
                            constraints.video = {{ facingMode: {{ ideal: _facing }} }};
                        }} else {{
                            constraints.video.facingMode = {{ ideal: _facing }};
                        }}
                    }}
                    return _orig(constraints);
                }};
            }}
        }})();
        </script>
        """, unsafe_allow_html=True)

        # ── Camera input — key berubah saat facing berubah agar kamera di-reinit ──
        image_input = st.camera_input(
            "Ambil foto daging sapi",
            key=f"camera_input_{facing_val}",
        )

    st.markdown("---")

    # Proses deteksi jika gambar diinput
    if image_input is not None:
        with st.spinner("Sedang memproses citra..."):
            try:
                # Proses gambar dari file upload atau camera input
                if isinstance(image_input, str):
                    # Seharusnya tidak terjadi lagi, tapi sebagai fallback
                    import base64 as _b64
                    _, encoded = image_input.split(",", 1)
                    img_bytes = _b64.b64decode(encoded)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:
                    img_bgr = process_image(image_input)

                if img_bgr is None:
                    st.error("Format berkas gambar tidak valid atau rusak.")
                    return

                # Jalankan prediksi model
                # conf=0.50: abaikan prediksi di bawah 50% confidence
                # iou=0.5: NMS hapus bounding box overlap > 50% (cegah double detection)
                results = model.predict(source=img_bgr, verbose=False, conf=0.50, iou=0.7)

                # Dapatkan hasil anotasi & filter confidence
                img_rgb, detections, max_conf, status = draw_boxes_and_filter(
                    img_bgr, results
                )

                # Tampilkan hasil deteksi
                if status == "error":
                    st.error(
                        "Tidak terdeteksi daging sapi yang valid. "
                        "Pastikan foto jelas, fokus pada objek, dan bukan gambar acak."
                    )
                    st.image(img_rgb, caption="Gambar Input", use_container_width=True)
                else:
                    if status == "warning":
                        st.warning(
                            "Tingkat keyakinan model rendah (di bawah 50%). Hasil deteksi mungkin kurang akurat. "
                            "Disarankan mengambil gambar ulang dengan pencahayaan yang lebih baik."
                        )
                    else:
                        st.success("Deteksi berhasil diselesaikan.")

                    # Layout hasil
                    col_view, col_info = st.columns([3, 2], gap="large")

                    with col_view:
                        st.image(img_rgb, caption="Citra Hasil Deteksi", use_container_width=True)

                    with col_info:
                        st.markdown("### Ringkasan Deteksi")

                        # Metrik Deteksi
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Objek Terdeteksi", len(detections))
                        with m2:
                            st.metric("Confidence Terkini", f"{max_conf:.1%}")

                        st.markdown("")
                        st.markdown("### Detail Hasil Klasifikasi")

                        for det in detections:
                            cls_name = det["class"]
                            conf = det["confidence"]
                            tag_cls = get_tag_class(cls_name)
                            bar_cls = get_conf_bar_class(conf)
                            icon = CLASS_ICONS.get(cls_name, "●")

                            st.markdown(f"""
                            <div class="detection-item">
                                <div>
                                    <span class="label-tag {tag_cls}"><span class="dot-icon">{icon}</span> {cls_name}</span>
                                </div>
                                <div class="det-conf">
                                    {conf:.1%}
                                    <div class="conf-bar-container">
                                        <div class="conf-bar {bar_cls}" style="width: {conf*100:.0f}%"></div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Tombol download hasil
                        st.markdown("")
                        img_pil = Image.fromarray(img_rgb)
                        buf = io.BytesIO()
                        img_pil.save(buf, format="PNG")
                        st.download_button(
                            label="Unduh Hasil Gambar",
                            data=buf.getvalue(),
                            file_name="scanmeat_result.png",
                            mime="image/png",
                            use_container_width=True,
                        )

                        # Disclaimer — di bawah download button, tetap dalam kolom kanan
                        st.markdown("")
                        st.markdown("""
                        <div class="info-card" style="border-left: 4px solid var(--warning); background-color: #fffbeb; padding: 1.25rem; margin-bottom: 0;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #a16207; font-weight: 700; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a16207" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                                Disclaimer Hasil Deteksi Model
                            </h4>
                            <p style="margin: 0; font-size: 0.8rem; color: #713f12; line-height: 1.5;">
                                Klasifikasi ini diproses secara otomatis oleh model deep learning YOLOv8s.
                                Akurasi hasil sangat bergantung pada kualitas foto, tingkat pencahayaan, dan sudut pengambilan gambar.
                                Gunakan hasil scan ini sebagai referensi awal, lalu konfirmasi melalui panduan ciri fisik daging di bawah.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                # Render Panduan Ciri Fisik Daging Sapi di bawah hasil scan
                render_beef_characteristics()

            except Exception as e:
                st.error(f"Terjadi kegagalan pemrosesan: {str(e)}")
    else:
        # Tampilan Awal/Kosong
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: var(--text-secondary);">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" style="display:inline">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
            </svg>
            <h3 style="font-size: 1.1rem; font-weight: 600; color: var(--text-primary); margin: 1rem 0 0.5rem 0;">Belum Ada Gambar yang Dianalisis</h3>
            <p style="font-size: 0.85rem; margin: 0;">Silakan unggah foto daging sapi atau gunakan kamera untuk melihat hasil klasifikasi.</p>
        </div>
        """, unsafe_allow_html=True)


def render_info_tab():
    """Merender konten untuk tab Tentang Sistem."""
    st.markdown("""
    <div class="info-card">
        <h3>Deskripsi Sistem</h3>
        <p>
            <strong>ScanMeat!</strong> adalah aplikasi berbasis web interaktif yang dikembangkan sebagai wadah implementasi model deteksi objek untuk klasifikasi kesegaran daging sapi.
            Sistem ini memanfaatkan arsitektur deep learning <strong>YOLOv8s</strong> dengan teknik <strong>Transfer Learning</strong> menggunakan pre-trained model.
        </p>
        <p>
            Model mendeteksi daging sapi secara real-time pada citra input dan secara otomatis memberikan klasifikasi berdasarkan tiga tingkat kesegaran: 
            <span class="class-badge tag-segar">Segar</span>, 
            <span class="class-badge tag-setengah">Setengah Segar</span>, dan 
            <span class="class-badge tag-busuk">Busuk</span>.
        </p>
    </div>

    <div class="info-card">
        <h3>Kebijakan Threshold & Mekanisme Validasi (Confidence)</h3>
        <p>
            Untuk meminimalkan kesalahan pembacaan objek acak atau citra non-daging, sistem menerapkan verifikasi berlapis berbasis tingkat kepercayaan (confidence score) model sebagai berikut:
        </p>
        <table class="threshold-table">
            <thead>
                <tr>
                    <th>Rentang Confidence</th>
                    <th>Status Validasi</th>
                    <th>Tindakan Sistem</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>&lt; 35%</strong> (Error Threshold)</td>
                    <td><span class="class-badge tag-busuk">Tidak Valid</span></td>
                    <td>Bounding box tidak digambar. Menampilkan pesan kesalahan berwarna merah untuk memfoto ulang dengan benar.</td>
                </tr>
                <tr>
                    <td><strong>35% &mdash; 50%</strong> (Warning Threshold)</td>
                    <td><span class="class-badge tag-setengah">Peringatan</span></td>
                    <td>Bounding box digambar. Menampilkan pesan peringatan berwarna kuning karena keakuratan model rendah.</td>
                </tr>
                <tr>
                    <td><strong>&gt; 50%</strong> (Success Threshold)</td>
                    <td><span class="class-badge tag-segar">Valid / Sukses</span></td>
                    <td>Bounding box digambar secara normal dan menampilkan informasi klasifikasi dengan sukses.</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="info-card">
        <h3>Panduan Pengambilan Gambar</h3>
        <p>Untuk mendapatkan hasil klasifikasi dengan akurasi maksimal, harap ikuti petunjuk berikut:</p>
        <ol style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6; padding-left: 1.25rem;">
            <li>Pastikan objek daging sapi berada di tengah frame dan dalam kondisi fokus (tidak buram).</li>
            <li>Gunakan pencahayaan yang cukup dan merata. Hindari bayangan gelap yang menutupi permukaan daging.</li>
            <li>Posisikan kamera sejajar atau mengarah langsung ke permukaan atas daging sapi.</li>
            <li>Pastikan tidak ada objek lain yang mendominasi gambar selain daging sapi yang ingin dideteksi.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Program ────────────────────────────────────────────────────────────

def main():
    inject_custom_css()
    render_top_navbar()

    # Load model YOLOv8s
    model, model_error = load_model()

    if model_error:
        st.error(f"Gagal memuat model: {model_error}")
        st.info("Pastikan file model diletakkan pada folder `models/best.pt`.")
        st.stop()

    # Navigasi Tab Utama (Berfungsi sebagai Navbar)
    tab_scan, tab_info = st.tabs(["Scan Daging", "Tentang Sistem"])

    with tab_scan:
        render_scan_tab(model)

    with tab_info:
        render_info_tab()

    # Footer
    st.markdown("""
    <div class="app-footer">
        ScanMeat! &mdash; Klasifikasi Tingkat Kesegaran Daging Sapi Berbasis YOLOv8s<br>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
