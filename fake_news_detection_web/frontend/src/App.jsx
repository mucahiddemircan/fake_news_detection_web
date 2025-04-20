import { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCheck = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Tahmin yapılamadı." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Fake News Detector</h1>
      <textarea
        placeholder="Haber metnini buraya yapıştırın..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={8}
        cols={60}
      />
      <br />
      <button onClick={handleCheck}>Haberi Kontrol Et</button>

      {loading && <p>Kontrol ediliyor...</p>}

      {result && !result.error && (
        <div className={`result ${result.result === "FAKE" ? "fake" : "real"}`}>
          <h2>
            Sonuç:{" "}
            {result.result === "FAKE" ? "🛑 Sahte Haber" : "✅ Gerçek Haber"}
          </h2>
          <p>Model Güveni: {(result.confidence * 100).toFixed(2)}%</p>
        </div>
      )}

      {result?.error && <p className="error">{result.error}</p>}
    </div>
  );
}

export default App;
