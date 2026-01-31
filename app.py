import streamlit as st
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="Teachable Machine Webcam",
    layout="centered"
)

# App title
st.title("📷 Teachable Machine – Webcam & Image Upload Classifier")
st.write("Use **Webcam** or **Upload an Image** to classify.")

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
        img {
            max-width: 300px;
            margin-top: 10px;
        }
    </style>
</head>
<body>

<h3>Teachable Machine Image Model</h3>

<!-- Webcam buttons -->
<button class="start" onclick="startWebcam()">▶ Start Webcam</button>
<button class="stop" onclick="stopWebcam()">⏹ Stop Webcam</button>

<br><br>

<!-- Upload image -->
<input type="file" accept="image/*" onchange="handleImageUpload(event)">
<br>

<div id="webcam-container"></div>
<img id="uploaded-image" />
<div id="label-container"></div>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

<script type="text/javascript">
    const URL = "https://teachablemachine.withgoogle.com/models/X17Jat3V8/";

    let model, webcam, labelContainer, maxPredictions;
    let isRunning = false;

  async function handleImageUpload(event) {
    await loadModel();   // Load model if not already loaded

    stopWebcam();        // Stop webcam if running

    const img = document.getElementById("uploaded-image");
    img.src = URL.createObjectURL(event.target.files[0]);
    img.style.display = "block";

    img.onload = async () => {
        await predict(img);  // ✅ Here the uploaded image is sent to the model
    };
}


    async function startWebcam() {
        if (isRunning) return;

        await loadModel();

        webcam = new tmImage.Webcam(224, 224, true);
        await webcam.setup();
        await webcam.play();
        isRunning = true;
        window.requestAnimationFrame(loop);

        document.getElementById("uploaded-image").style.display = "none";
        document.getElementById("webcam-container").innerHTML = "";
        document.getElementById("webcam-container").appendChild(webcam.canvas);
    }

    async function loop() {
        if (!isRunning) return;
        webcam.update();
        await predict(webcam.canvas);
        window.requestAnimationFrame(loop);
    }

    async function predict(imageSource) {
        const prediction = await model.predict(imageSource);
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
        labelContainer.innerHTML = "<em>Webcam stopped</em>";
    }

    async function handleImageUpload(event) {
        await loadModel();

        stopWebcam();

        const img = document.getElementById("uploaded-image");
        img.src = URL.createObjectURL(event.target.files[0]);
        img.style.display = "block";

        img.onload = async () => {
            await predict(img);
        };
    }
</script>

</body>
</html>
"""

components.html(html_code, height=750)
