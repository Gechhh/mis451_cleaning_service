import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
from io import BytesIO

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Teachable Machine",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Teachable Machine")
st.caption("Minimal image classification demo")

# --------------------------------------------------
# Shared CSS snippet (prediction rows) — reused in both tabs
# --------------------------------------------------
SHARED_ROW_CSS = """
    .result-row {
        display: flex;
        align-items: center;
        position: relative;
        overflow: hidden;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: default;
    }

    .result-row:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.13);
    }

    /* animated background fill */
    .result-row .bar-fill {
        position: absolute;
        top: 0; left: 0;
        height: 100%;
        z-index: 0;
        opacity: 0.15;
        border-radius: 12px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* text layer sits above the bar */
    .result-row .content {
        position: relative;
        z-index: 1;
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }

    .result-row .class-name {
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .result-row .confidence {
        font-size: 15px;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }

    /* top-pick border + badge */
    .result-row.top-pick {
        border: 2px solid currentColor;
    }

    .result-row.top-pick .badge {
        display: inline-block;
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: currentColor;
        color: white;
        padding: 2px 8px;
        border-radius: 20px;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* ---- theme colours ---- */
    .theme-clean            { background: #dbeafe; color: #1e40af; }
    .theme-clean .bar-fill  { background: #3b82f6; }

    .theme-messy            { background: #fef9c3; color: #854d0e; }
    .theme-messy .bar-fill  { background: #eab308; }

    .theme-default            { background: #f3f4f6; color: #111827; }
    .theme-default .bar-fill  { background: #6b7280; }
"""

# Shared JS: renders prediction array → styled rows
SHARED_RENDER_JS = """
    function renderPredictions(prediction, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = "";

        prediction
            .sort((a, b) => b.probability - a.probability)
            .forEach((p, index) => {
                const row = document.createElement("div");
                row.className = "result-row";

                // pick theme
                const nameLower = p.className.toLowerCase();
                if      (nameLower.includes("clean")) row.classList.add("theme-clean");
                else if (nameLower.includes("messy")) row.classList.add("theme-messy");
                else                                  row.classList.add("theme-default");

                // highlight top result
                if (index === 0) row.classList.add("top-pick");

                const pct = (p.probability * 100).toFixed(1);

                // background bar
                const barFill = document.createElement("div");
                barFill.className = "bar-fill";
                barFill.style.width = pct + "%";
                row.appendChild(barFill);

                // text content
                const badge = index === 0 ? '<span class="badge">Best Match</span>' : '';
                row.insertAdjacentHTML("beforeend", `
                    <div class="content">
                        <span class="class-name">${p.className}${badge}</span>
                        <span class="confidence">${pct}%</span>
                    </div>
                `);

                container.appendChild(row);
            });
    }
"""

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2 = st.tabs(["📷 Webcam", "📁 Upload Image"])

# ==================================================
# TAB 1 — WEBCAM
# ==================================================
with tab1:
    st.write("Start the webcam to classify objects in real time.")

    webcam_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #fafafa;
                margin: 0;
                padding: 24px;
            }}

            .card {{
                background: white;
                border-radius: 16px;
                padding: 28px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.07);
                max-width: 440px;
                margin: auto;
                text-align: center;
            }}

            h3 {{
                margin-top: 0;
                margin-bottom: 20px;
                font-size: 18px;
                color: #1f2937;
            }}

            .btn-row {{
                display: flex;
                justify-content: center;
                gap: 10px;
                margin-bottom: 20px;
            }}

            button {{
                padding: 10px 22px;
                font-size: 14px;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                font-weight: 600;
                transition: background 0.2s, transform 0.1s;
            }}

            button:active {{ transform: scale(0.96); }}

            button.btn-start {{
                background: #86efac;
                color: #065f46;
            }}
            button.btn-start:hover {{ background: #4ade80; }}

            button.btn-stop {{
                background: #e5e7eb;
                color: #374151;
            }}
            button.btn-stop:hover {{ background: #d1d5db; }}

            #webcam-container canvas {{
                border-radius: 10px;
                max-width: 100%;
            }}

            #label-container {{ margin-top: 18px; }}

            .muted {{
                color: #9ca3af;
                font-size: 12px;
                margin-top: 14px;
            }}

            {SHARED_ROW_CSS}
        </style>
    </head>
    <body>
        <div class="card">
            <h3>📷 Webcam Prediction</h3>

            <div class="btn-row">
                <button class="btn-start" onclick="startWebcam()">▶ Start</button>
                <button class="btn-stop"  onclick="stopWebcam()">⏹ Stop</button>
            </div>

            <div id="webcam-container"></div>
            <div id="label-container"></div>
            <div class="muted">Allow camera access when prompted</div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>
        <script>
            {SHARED_RENDER_JS}

            const URL = "https://teachablemachine.withgoogle.com/models/X17Jat3V8/";
            let model, webcam;
            let running = false;
            let frameCount = 0;

            async function startWebcam() {{
                if (running) return;

                model = await tmImage.load(URL + "model.json", URL + "metadata.json");

                webcam = new tmImage.Webcam(224, 224, true);
                await webcam.setup();
                await webcam.play();

                running = true;
                frameCount = 0;

                const wc = document.getElementById("webcam-container");
                wc.innerHTML = "";
                wc.appendChild(webcam.canvas);

                document.getElementById("label-container").innerHTML = "";
                loop();
            }}

            async function loop() {{
                if (!running) return;
                webcam.update();
                frameCount++;
                if (frameCount % 10 === 0) {{
                    const prediction = await model.predict(webcam.canvas);
                    renderPredictions(prediction, "label-container");
                }}
                requestAnimationFrame(loop);
            }}

            function stopWebcam() {{
                running = false;
                if (webcam) webcam.stop();
                document.getElementById("webcam-container").innerHTML = "";
                document.getElementById("label-container").innerHTML =
                    '<div class="muted">Webcam stopped</div>';
            }}
        </script>
    </body>
    </html>
    """

    components.html(webcam_html, height=580)

# ==================================================
# TAB 2 — IMAGE UPLOAD
# ==================================================
with tab2:
    st.write("Upload an image to classify it using your Teachable Machine model.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)

        # base64 encode once
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # ── two-column layout ──
        col1, col2 = st.columns([1.1, 1], gap="medium")

        # ── LEFT: uploaded image ──
        with col1:
            st.image(image, use_container_width=True, caption="Uploaded Image")

        # ── RIGHT: prediction card ──
        with col2:
            upload_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                        background: #fafafa;
                        margin: 0;
                        padding: 20px;
                    }}

                    .card {{
                        background: white;
                        border-radius: 16px;
                        padding: 28px;
                        box-shadow: 0 4px 16px rgba(0,0,0,0.07);
                    }}

                    h3 {{
                        margin-top: 0;
                        margin-bottom: 22px;
                        font-size: 18px;
                        color: #1f2937;
                        text-align: center;
                    }}

                    .loading {{
                        text-align: center;
                        color: #9ca3af;
                        font-size: 14px;
                        padding: 24px 0;
                    }}

                    .spinner {{
                        border: 3px solid #e5e7eb;
                        border-top: 3px solid #6366f1;
                        border-radius: 50%;
                        width: 28px;
                        height: 28px;
                        animation: spin 0.8s linear infinite;
                        margin: 0 auto 12px;
                    }}

                    @keyframes spin {{
                        to {{ transform: rotate(360deg); }}
                    }}

                    {SHARED_ROW_CSS}
                </style>
            </head>
            <body>
                <div class="card">
                    <h3>🎯 Prediction Results</h3>
                    <div id="label-container">
                        <div class="loading">
                            <div class="spinner"></div>
                            Analyzing image…
                        </div>
                    </div>
                </div>

                <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>
                <script>
                    {SHARED_RENDER_JS}

                    const URL = "https://teachablemachine.withgoogle.com/models/X17Jat3V8/";

                    async function predictImage() {{
                        const model = await tmImage.load(URL + "model.json", URL + "metadata.json");

                        const img = new Image();
                        img.src = "data:image/png;base64,{img_str}";
                        await new Promise(resolve => img.onload = resolve);

                        const prediction = await model.predict(img);
                        renderPredictions(prediction, "label-container");
                    }}

                    predictImage();
                </script>
            </body>
            </html>
            """

            components.html(upload_html, height=380)

    else:
        st.info("👆 Upload an image to get started!")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Both modes use the same Teachable Machine model for predictions.")