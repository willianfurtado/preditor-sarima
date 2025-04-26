const tf = require("@tensorflow/tfjs");
const ARIMA = require("arima");

// ====== Configurações da Google Sheets API ======
const API_KEY = "AIzaSyD36C-k9xtxWnkzTv5RxZIf-rqyAtLWed4";
const SPREADSHEET_ID = "1cTcdqJtk1qfWeCmfOAKkLae2vZfEd8rYLWzyQWPTb4A";
const RANGE = "B3:B1000";
const url = `https://sheets.googleapis.com/v4/spreadsheets/${SPREADSHEET_ID}/values/${RANGE}?key=${API_KEY}`;

// Função principal
(async () => {
  // ====== Pega os dados da planilha ======
  const response = await fetch(url);
  const data = await response.json();

  //   console.log(data)

  if (!data.values || data.values.length === 0) {
    console.error("Nenhum dado encontrado na planilha.");
    return;
  }

  const ts = data.values
    .map((linha) => {
      if (linha[0] && typeof linha[0] === "string") {
        let valorNumerico = linha[0].replace(",", ".");
        if (!isNaN(valorNumerico)) {
          return parseFloat(valorNumerico);
        } else {
          console.error("Erro: valor inválido detectado: ", linha[0]);
          return null;
        }
      } else {
        console.error("Erro: valor inválido detectado: ", linha[0]);
        return null;
      }
    })
    .filter((valor) => valor != null);

  // ====== SARIMA ======
  const arima = new ARIMA({
    p: 3,
    d: 1,
    q: 3,
    P: 1,
    D: 1,
    Q: 0,
    s: 5,
    verbose: false,
  }).train(ts);

  const [sarimaPred] = arima.predict(1);
  console.log("SARIMA prevê:", sarimaPred[0].toFixed(2));

  const extendedTs = [...ts, sarimaPred[0]];

  // ====== Normalização ======
  const max = Math.max(...extendedTs);
  const min = Math.min(...extendedTs);
  const normalize = (x) => (x - min) / (max - min);
  const denormalize = (x) => x * (max - min) + min;
  const normalized = extendedTs.map(normalize);

  // ====== Prepara janelas para LSTM ======
  const window = 5;
  const xs = [];
  const ys = [];

  for (let i = 0; i < normalized.length - window; i++) {
    const inputWindow = normalized.slice(i, i + window).map((v) => [v]);
    const outputValue = [normalized[i + window]];
    xs.push(inputWindow);
    ys.push(outputValue);
  }

  const xsTensor = tf.tensor3d(xs);
  const ysTensor = tf.tensor2d(ys);

  // ====== Modelo LSTM ======
  const model = tf.sequential();
  model.add(
    tf.layers.lstm({
      units: 50,
      inputShape: [window, 1],
      returnSequences: false,
    })
  );
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ loss: "meanSquaredError", optimizer: "adam" });

  console.log("Treinando modelo LSTM...");
  await model.fit(xsTensor, ysTensor, {
    epochs: 100,
    batchSize: 8,
    verbose: 0,
  });

  // ====== Previsão final híbrida ======
  const inputLSTM = tf.tensor3d([normalized.slice(-window).map((v) => [v])]);
  const predLSTM = model.predict(inputLSTM);
  const predValue = predLSTM.dataSync()[0];

  console.log("LSTM com dado SARIMA prevê:", denormalize(predValue).toFixed(2));
})();
