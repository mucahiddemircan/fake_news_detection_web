import { useState } from "react";
import "../styles/LeftBar.scss";

const LeftSide = ({ value }) => {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCheck = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(
        `http://127.0.0.1:5000/${value.selectedModel}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text }),
        }
      );

      const data = await response.json();
      setResult(data);
      setLoading(false);
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Tahmin yapılamadı." });
      setLoading(false);
    }
  };

  return (
    <div className="leftside-container">
      {/* <div className="selected-value">
                {value.layer && value.units && (
                    <p>
                        Seçtiğiniz <b>{value?.selectedModel.toUpperCase()}</b> modelde Units değerini '{value?.units}' olarak belirlediniz.
                    </p>
                )}
                {value.layerActivition1 && (
                    <p>
                        Modelin birinci katman Activation foknksiyonu : {value.layerActivition1.toUpperCase()}
                    </p>
                )}
                {value.layerActivition2 && (
                    <p>
                        Modelin birinci katman Activation foknksiyonu : {value.layerActivition2.toUpperCase()}
                    </p>
                )}
                {value.layerActivition3 && (
                    <p>
                        Modelin birinci katman Activation foknksiyonu : {value.layerActivition3.toUpperCase()}
                    </p>
                )}
            </div> */}
      <div className="left-bar">
        <p className="header-text">Fake News Detector</p>
        <hr />
        <textarea
          placeholder="Haber metnini buraya giriniz..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={8}
          cols={60}
        />
        <br />
        <button onClick={handleCheck}>Haberi Kontrol Et</button>

        {loading && <p>Kontrol ediliyor...</p>}

        {result && !result.error && (
          <div
            className={`result ${result.result === "FAKE" ? "fake" : "real"}`}
          >
            <h2>
              Sonuç:{" "}
              {result.result === "FAKE" ? "🛑 Sahte Haber" : "✅ Gerçek Haber"}
            </h2>
          </div>
        )}

        {result?.error && <p className="error">{result.error}</p>}
      </div>
    </div>
  );
};

export default LeftSide;
