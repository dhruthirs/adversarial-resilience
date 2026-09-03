import { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [model, setModel] = useState("ResNet18");
  const [attack, setAttack] = useState("FGSM");

  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (file) {
      setImage(URL.createObjectURL(file));
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    console.log("Selected model:", model);
    console.log("Selected attack:", attack);
    console.log("Selected image:", image);

    // Backend connection will be added later.
  };

  return (
    <div className="app">
      <header>
        <h1>Adversarial Resilience Analyzer</h1>
        <p>Analyze the impact of adversarial attacks on deep learning models</p>
      </header>

      <main>
        <section className="upload-section">
          <h2>Upload Image</h2>

          <label className="upload-box">
            <input
              type="file"
              accept="image/png, image/jpeg, image/jpg"
              onChange={handleImageChange}
            />

            {image ? (
              <img src={image} alt="Uploaded preview" />
            ) : (
              <p>Click to choose an image</p>
            )}
          </label>
        </section>

        <section className="controls">
          <div className="control">
            <label>Model</label>

            <select
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              <option value="ResNet18">ResNet18</option>
              <option value="BasicANN">Basic ANN</option>
            </select>
          </div>

          <div className="control">
            <label>Attack</label>

            <select
              value={attack}
              onChange={(event) => setAttack(event.target.value)}
            >
              <option value="FGSM">FGSM</option>
              <option value="PGD">PGD</option>
            </select>
          </div>
        </section>

        <button
          className="analyze-button"
          onClick={handleSubmit}
          disabled={!image}
        >
          Run Analysis
        </button>
      </main>
    </div>
  );
}

export default App;