import streamlit as st
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="Teachable Machine Webcam",
    layout="centered"
)

# App title
st.title("📷 Teachable Machine – Webcam Classifier")
st.write("Click **Start** to turn on the webcam and **Stop** to turn it off.")

# HTML + JavaScript
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

<h3>Teachable Machine Image Model</h3>

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

# Render HTML inside Streamlit
components.html(html_code, height=600)
