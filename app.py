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
    layout="centered"
)

st.title("Teachable Machine")
st.caption("Minimal image classification demo")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2 = st.tabs(["Webcam", "Upload Image"])

# ==================================================
# TAB 1 — WEBCAM
# ==================================================
with tab1:
    st.write("Start the webcam to classify objects in real time.")

    webcam_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">

        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #fafafa;
                margin: 0;
                padding: 20px;
            }

            .card {
                background: white;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.06);
                max-width: 420px;
                margin: auto;
                text-align: center;
            }

            h3 {
                margin-top: 0;
                margin-bottom: 20px;
            }

            /* START BUTTON */
            button {
                padding: 10px 18px;
                font-size: 14px;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                margin: 6px;
                background: #86efac;
                color: #065f46;
                font-weight: 600;
            }

            button:hover {
                background: #6ee7b7;
            }

            /* STOP BUTTON */
            button.secondary {
                background: #e5e7eb;
                color: #111827;
            }

            #label-container div {
                padding: 10px 14px;
                margin: 8px 0;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                font-size: 14px;
                background: #f3f4f6;
            }

            .confidence {
                font-weight: 600;
            }

            .muted {
                color: #6b7280;
                font-size: 13px;
                margin-top: 12px;
            }
        </style>
    </head>

    <body>
        <div class="card">
            <h3>Webcam Prediction</h3>

            <button onclick="startWebcam()">Start</button>
            <button class="secondary" onclick="stopWebcam()">Stop</button>

            <div id="webcam-container" style="margin-top:20px;"></div>
            <div id="label-container"></div>

            <div class="muted">Allow camera access when prompted</div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

        <script>
            const URL = "https://teachablemachine.withgoogle.com/models/X17Jat3V8/";

            let model, webcam, labelContainer;
            let running = false;
            let frameCount = 0;

            async function startWebcam() {
                if (running) return;

                model = await tmImage.load(
                    URL + "model.json",
                    URL + "metadata.json"
                );

                webcam = new tmImage.Webcam(224, 224, true);
                await webcam.setup();
                await webcam.play();

                running = true;
                frameCount = 0;
                document.getElementById("webcam-container").innerHTML = "";
                document.getElementById("webcam-container").appendChild(webcam.canvas);

                labelContainer = document.getElementById("label-container");
                labelContainer.innerHTML = "";

                loop();
            }

            async function loop() {
                if (!running) return;

                webcam.update();
                frameCount++;

                // Predict every 10 frames (~6 times per second)
                if (frameCount % 10 === 0) {
                    await predict();
                }

                requestAnimationFrame(loop);
            }

            async function predict() {
                const prediction = await model.predict(webcam.canvas);
                labelContainer.innerHTML = "";

                prediction
                    .sort((a, b) => b.probability - a.probability)
                    .forEach(p => {
                        const row = document.createElement("div");

                        let bg = "#f3f4f6";
                        let color = "#111827";

                        if (p.className.toLowerCase().includes("clean")) {
                            bg = "#dbeafe";
                            color = "#1e40af";
                        }

                        if (p.className.toLowerCase().includes("messy")) {
                            bg = "#fef9c3";
                            color = "#854d0e";
                        }

                        row.style.background = bg;
                        row.style.color = color;

                        row.innerHTML = `
                            <span>${p.className}</span>
                            <span class="confidence">${(p.probability * 100).toFixed(1)}%</span>
                        `;

                        labelContainer.appendChild(row);
                    });
            }

            function stopWebcam() {
                running = false;
                if (webcam) webcam.stop();
                document.getElementById("webcam-container").innerHTML = "";
                document.getElementById("label-container").innerHTML =
                    "<div class='muted'>Webcam stopped</div>";
            }
        </script>
    </body>
    </html>
    """

    components.html(webcam_html, height=560)

# ==================================================
# TAB 2 — IMAGE UPLOAD
# ==================================================
with tab2:
    st.write("Upload an image to classify it.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            upload_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                        background: #fafafa;
                        padding: 20px;
                    }}

                    .card {{
                        background: white;
                        border-radius: 12px;
                        padding: 24px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
                        max-width: 360px;
                        margin: auto;
                    }}

                    #label-container div {{
                        padding: 10px 14px;
                        margin: 8px 0;
                        border-radius: 8px;
                        display: flex;
                        justify-content: space-between;
                        font-size: 14px;
                    }}.confidence {{
                        font-weight: 600;
                    }}
                </style>
            </head>

            <body>
                <div class="card">
                    <h3 style="text-align:center;">Prediction</h3>
                    <div id="label-container">Analyzing…</div>
                </div>

                <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

                <script>
                    const URL = "https://teachablemachine.withgoogle.com/models/X17Jat3V8/";

                    async function predictImage() {{
                        const model = await tmImage.load(
                            URL + "model.json",
                            URL + "metadata.json"
                        );

                        const img = new Image();
                        img.src = "data:image/png;base64,{img_str}";
                        await new Promise(resolve => img.onload = resolve);

                        const prediction = await model.predict(img);
                        const container = document.getElementById("label-container");
                        container.innerHTML = "";

                        prediction
                            .sort((a, b) => b.probability - a.probability)
                            .forEach(p => {{
                                const row = document.createElement("div");

                                let bg = "#f3f4f6";
                                let color = "#111827";

                                if (p.className.toLowerCase().includes("clean")) {{
                                    bg = "#dbeafe";
                                    color = "#1e40af";
                                }}

                                if (p.className.toLowerCase().includes("messy")) {{
                                    bg = "#fef9c3";
                                    color = "#854d0e";
                                }}

                                row.style.background = bg;
                                row.style.color = color;

                                row.innerHTML = `
                                    <span>${{p.className}}</span>
                                    <span class="confidence">${{(p.probability * 100).toFixed(1)}}%</span>
                                `;

                                container.appendChild(row);
                            }});
                    }}

                    predictImage();
                </script>
            </body>
            </html>
            """

            components.html(upload_html, height=460)

    else:
        st.info("Upload an image to begin")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Uses the same Teachable Machine model for webcam and uploads.")