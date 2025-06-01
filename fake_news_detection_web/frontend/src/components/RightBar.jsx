import React, { useState } from "react";
import "../styles/rightBar.scss";
import LeftSide from "./LeftSide";

const RightBar = () => {
  const [selectedModel, setSelectedModel] = useState("");

  const selectedValue = {
    selectedModel,
  };

  console.log(selectedModel);

  return (
    <div className="container">
      <div className="right-container">
        <div className="select-model">
          <label className="header">Model Seçiniz:</label>
          <div className="radio-input">
            <label>
              <input
                type="radio"
                name="model"
                value="rnn"
                checked={selectedModel === "rnn"}
                onChange={(e) => setSelectedModel(e.target.value)}
              />
              <span>RNN</span>
            </label>
            <label>
              <input
                type="radio"
                name="model"
                value="lstm"
                checked={selectedModel === "lstm"}
                onChange={(e) => setSelectedModel(e.target.value)}
              />
              <span>LSTM</span>
            </label>
            <label>
              <input
                type="radio"
                name="model"
                value="gru"
                checked={selectedModel === "gru"}
                onChange={(e) => setSelectedModel(e.target.value)}
              />
              <span>GRU</span>
            </label>
            <label>
              <input
                type="radio"
                name="model"
                value="bert"
                checked={selectedModel === "bert"}
                onChange={(e) => setSelectedModel(e.target.value)}
              />
              <span>BERT</span>
            </label>
            <span className="selection"></span>
          </div>
        </div>
      </div>
      <LeftSide value={selectedValue} />
    </div>
  );
};

export default RightBar;
