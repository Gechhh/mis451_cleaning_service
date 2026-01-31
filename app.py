import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Teachable Machine Classifier",
    layout="centered"
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
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # HTML for image prediction
        upload_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    text-align: center;
                }}
                #prediction-container {{
                    margin-top: 20px;
                }}
                #label-container div {{
                    font-size: 18px;
                    margin: 8px 0;
                    padding: 10px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                }}
                .loading {{
                    color: #666;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>

        <div id="prediction-container">
            <h3>Predictions</h3>
            <div id="label-container" class="loading">Loading model and analyzing image...</div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

        <script type="text/javascript">
            const URL = "https://teachablemachine.withgoogle.com/models/X17Jat3V8/";
            
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
                    div.innerHTML = prediction[i].className + ": " + percentage + "%";
                    
                    // Highlight the top prediction
                    if (i === 0) {{
                        div.style.backgroundColor = "#4CAF50";
                        div.style.color = "white";
                        div.style.fontWeight = "bold";
                    }}
                    
                    labelContainer.appendChild(div);
                }}
            }}
            
            predictImage();
        </script>

        </body>
        </html>
        """
        
        components.html(upload_html, height=400)
    else:
        st.info("👆 Upload an image to get started!")

# Add footer
st.markdown("---")
st.markdown("**Note:** Both modes use the same Teachable Machine model for predictions.")