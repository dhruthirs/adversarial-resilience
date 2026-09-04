import { useState } from "react";
import "./App.css";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [selectedModel, setSelectedModel] = useState("ANN");
  const [selectedAttack, setSelectedAttack] = useState("FGSM");

  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const runAnalysis = async () => {
    if (!selectedImage) return;

    const formData = new FormData();

    formData.append("file", selectedImage);
    formData.append("model", selectedModel);
    formData.append("attack", selectedAttack);

    try {
      setLoading(true);
      setResult(null);

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      setResult(data);
    } catch (error) {
      console.error(error);

      setResult({
        message: "Could not connect to backend.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      {/* Header */}
      <header>
        <h1>Adversarial Resilience Analyzer</h1>

        <p>
          Analyze how machine learning models respond to adversarial attacks.
        </p>
      </header>


      <main>

        {/* Image Upload Section */}
        <div className="upload-section">

          <h2>Upload Image</h2>

          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
          />

          {preview && (
            <div className="preview">
              <img
                src={preview}
                alt="Selected"
              />
            </div>
          )}

        </div>


        {/* Model and Attack Controls */}
        <div className="controls">

          <label>
            Model:

            <select
              value={selectedModel}
              onChange={(event) =>
                setSelectedModel(event.target.value)
              }
            >
              <option>ANN</option>
              <option>CNN</option>
              <option>LSTM</option>
              <option>BiLSTM</option>
              <option>KAN</option>
              <option>ViT</option>
            </select>
          </label>


          <label>
            Attack:

            <select
              value={selectedAttack}
              onChange={(event) =>
                setSelectedAttack(event.target.value)
              }
            >
              <option>FGSM</option>
              <option>PGD</option>
            </select>
          </label>

        </div>


        {/* Run Analysis Button */}
        <button
          onClick={runAnalysis}
          disabled={!selectedImage || loading}
        >
          {loading ? "Analyzing..." : "Run Analysis"}
        </button>


        {/* Result Section */}
        {result && (
          <div className="result">

            <h2>Analysis Result</h2>

            {result.predicted_digit !== undefined && (
              <>
                <p>
                  <strong>Predicted Digit:</strong>{" "}
                  {result.predicted_digit}
                </p>

                <p>
                  <strong>Confidence:</strong>{" "}
                  {(result.confidence * 100).toFixed(2)}%
                </p>
              </>
            )}

            {result.message && (
              <p>
                <strong>Message:</strong>{" "}
                {result.message}
              </p>
            )}

          </div>
        )}

      </main>

    </div>
  );
}

export default App;