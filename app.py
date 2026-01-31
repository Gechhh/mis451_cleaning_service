import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Teachable Machine Classifier",
    layout="wide"
)

# App title
st.title("🎯 Teachable Machine – Image Classifier")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📷 Webcam", "📁 Upload Image"])

with tab1:
    st.write("Click **Start** to turn on the webcam and **Stop** to turn it off.")
    
    # HTML + JavaScript for webcam
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Teachable Machine Image Model</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
            }
            button {
                padding: 10px 22px;
                font-size: 16px;
                margin: 10px;
                cursor: pointer;
                border-radius: 6px;
                border: none;
            }
            .start {
                background-color: #2ecc71;
                color: white;
            }
            .stop {
                background-color: #e74c3c;
                color: white;
            }
            #label-container div {
                font-size: 18px;
                margin: 6px 0;
            }
        </style>
    </head>
    <body>

    <h3>Webcam Prediction</h3>

    <button class="start" onclick="startWebcam()">▶ Start</button>
    <button class="stop" onclick="stopWebcam()">⏹ Stop</button>

    <div id="webcam-container"></div>
    <div id="label-container"></div>

    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

    <script type="text/javascript">
        const URL = "https://teachablemachine.withgoogle.com/models/X17Jat3V8/";

        let model, webcam, labelContainer, maxPredictions;
        let isRunning = false;

        async function startWebcam() {
            if (isRunning) return;

            const modelURL = URL + "model.json";
            const metadataURL = URL + "metadata.json";

            model = await tmImage.load(modelURL, metadataURL);
            maxPredictions = model.getTotalClasses();

            webcam = new tmImage.Webcam(224, 224, true);
            await webcam.setup();
            await webcam.play();

            isRunning = true;
            window.requestAnimationFrame(loop);

            document.getElementById("webcam-container").innerHTML = "";
            document.getElementById("webcam-container").appendChild(webcam.canvas);

            labelContainer = document.getElementById("label-container");
            labelContainer.innerHTML = "";

            for (let i = 0; i < maxPredictions; i++) {
                labelContainer.appendChild(document.createElement("div"));
            }
        }

        async function loop() {
            if (!isRunning) return;

            webcam.update();
            await predict();
            window.requestAnimationFrame(loop);
        }

        async function predict() {
            const prediction = await model.predict(webcam.canvas);
            for (let i = 0; i < maxPredictions; i++) {
                labelContainer.childNodes[i].innerHTML =
                    prediction[i].className + ": " +
                    prediction[i].probability.toFixed(2);
            }
        }

        function stopWebcam() {
            if (!isRunning) return;

            isRunning = false;
            webcam.stop();

            document.getElementById("webcam-container").innerHTML = "";
            document.getElementById("label-container").innerHTML =
                "<em>Webcam stopped</em>";
        }
    </script>

    </body>
    </html>
    """
    
    components.html(html_code, height=600)

with tab2:
    st.write("Upload an image to classify it using your Teachable Machine model.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # HTML for image prediction with better styling
            upload_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                    }}
                    
                    .container {{
                        background: white;
                        border-radius: 15px;
                        padding: 30px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    }}
                    
                    h3 {{
                        color: #333;
                        margin-top: 0;
                        margin-bottom: 25px;
                        font-size: 24px;
                        text-align: center;
                        border-bottom: 3px solid #667eea;
                        padding-bottom: 15px;
                    }}
                    
                    #label-container {{
                        margin-top: 20px;
                    }}
                    
                    #label-container div {{
                        font-size: 16px;
                        margin: 12px 0;
                        padding: 15px 20px;
                        border-radius: 10px;
                        transition: all 0.3s ease;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    
                    #label-container div:hover {{
                        transform: translateX(5px);
                    }}
                    
                    .loading {{
                        color: #666;
                        font-style: italic;
                        text-align: center;
                        padding: 30px;
                        background: #f8f9fa;
                        border-radius: 10px;
                    }}
                    
                    .spinner {{
                        border: 3px solid #f3f3f3;
                        border-top: 3px solid #667eea;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        animation: spin 1s linear infinite;
                        margin: 20px auto;
                    }}
                    
                    @keyframes spin {{
                        0% {{ transform: rotate(0deg); }}
                        100% {{ transform: rotate(360deg); }}
                    }}
                    
                    .class-name {{
                        font-weight: 600;
                        color: #333;
                    }}
                    
                    .percentage {{
                        font-weight: bold;
                        font-size: 18px;
                    }}
                    
                    .progress-bar {{
                        height: 8px;
                        background: #e0e0e0;
                        border-radius: 10px;
                        overflow: hidden;
                        margin-top: 8px;
                    }}
                    
                    .progress-fill {{
                        height: 100%;
                        border-radius: 10px;
                        transition: width 0.5s ease;
                    }}
                </style>
            </head>
            <body>

            <div class="container">
                <h3>🎯 Prediction Results</h3>
                <div id="label-container" class="loading">
                    <div class="spinner"></div>
                    Analyzing image...
                </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

            <script type="text/javascript">
                const URL = "https://teachablemachine.withgoogle.com/models/X17Jat3V8/";
                
                // Color palette for predictions
                const colors = [
                    {{bg: '#4CAF50', text: '#ffffff'}},  // Green for top
                    {{bg: '#2196F3', text: '#ffffff'}},  // Blue
                    {{bg: '#FF9800', text: '#ffffff'}},  // Orange
                    {{bg: '#9C27B0', text: '#ffffff'}},  // Purple
                    {{bg: '#607D8B', text: '#ffffff'}}   // Blue Grey
                ];
                
                async function predictImage() {{
                    const modelURL = URL + "model.json";
                    const metadataURL = URL + "metadata.json";
                    
                    const model = await tmImage.load(modelURL, metadataURL);
                    const maxPredictions = model.getTotalClasses();
                    
                    // Create image element from base64
                    const img = new Image();
                    img.src = "data:image/png;base64,{img_str}";
                    
                    await new Promise((resolve) => {{
                        img.onload = resolve;
                    }});
                    
                    // Make prediction
                    const prediction = await model.predict(img);
                    
                    // Display results
                    const labelContainer = document.getElementById("label-container");
                    labelContainer.innerHTML = "";
                    labelContainer.className = "";
                    
                    // Sort predictions by probability (highest first)
                    prediction.sort((a, b) => b.probability - a.probability);
                    
                    for (let i = 0; i < maxPredictions; i++) {{
                        const div = document.createElement("div");
                        const percentage = (prediction[i].probability * 100).toFixed(1);
                        const color = colors[i % colors.length];
                        
                        div.style.background = color.bg;
                        div.style.color = color.text;
                        
                        if (i === 0) {{
                            div.style.boxShadow = '0 5px 15px rgba(76, 175, 80, 0.4)';
                            div.style.transform = 'scale(1.02)';
                        }}
                        
                        div.innerHTML = `
                            <span class="class-name">${{prediction[i].className}}</span>
                            <span class="percentage">${{percentage}}%</span>
                        `;
                        
                        // Add progress bar
                        const progressBar = document.createElement("div");
                        progressBar.className = "progress-bar";
                        progressBar.style.marginTop = "10px";
                        
                        const progressFill = document.createElement("div");
                        progressFill.className = "progress-fill";
                        progressFill.style.width = percentage + "%";
                        progressFill.style.background = "rgba(255, 255, 255, 0.5)";
                        
                        progressBar.appendChild(progressFill);
                        div.appendChild(progressBar);
                        
                        labelContainer.appendChild(div);
                    }}
                }}
                
                predictImage();
            </script>

            </body>
            </html>
            """
            
            components.html(upload_html, height=600, scrolling=False)
    else:
        st.info("👆 Upload an image to get started!")

# Add footer
st.markdown("---")
st.markdown("**Note:** Both modes use the same Teachable Machine model for predictions.")