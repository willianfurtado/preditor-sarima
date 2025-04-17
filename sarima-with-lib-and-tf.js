const fs = require("fs");
const tf = require("@tensorflow/tfjs");
const parse = require("csv-parse/sync");
const ARIMA = require("arima");

// ====== Carrega e prepara os dados ======
const file = fs.readFileSync("./Cotacoes_Filtradas_nov_abril.csv");
const records = parse.parse(file, {
  columns: true,
  skip_empty_lines: true,
  delimiter: ",",
});

//padronização dos dados pro SARIMA (ordenando do mais antigo para o mais recente)
const ts = records.map((r) => parseFloat(r["Último"])).reverse();

// ====== Treinando o SARIMA ======
const arima = new ARIMA({
  p: 3,
  d: 1,
  q: 3,
  P: 1,
  D: 1,
  Q: 0,
  s: 5,
  verbose: false,
}).train(ts); //R$328.0

const [sarimaPred] = arima.predict(1); //Previsão para amanhã
console.log("SARIMA prevê:", sarimaPred[0].toFixed(2));

// ====== Insere a previsão do SARIMA como ponto futuro ======
const extendedTs = [...ts, sarimaPred[0]];

// ====== Normalização dos dados para LSTM ======
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

// ====== Define e treina o modelo LSTM ======
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

(async () => {
  console.log("Treinando modelo LSTM...");
  await model.fit(xsTensor, ysTensor, {
    epochs: 100,
    batchSize: 8,
    verbose: 0,
  });

  // === 7. Previsão com o modelo híbrido ===
  const inputLSTM = tf.tensor3d([normalized.slice(-window).map((v) => [v])]); // última janela com valor do SARIMA
  const predLSTM = model.predict(inputLSTM);
  const predValue = predLSTM.dataSync()[0];

  console.log(
    "LSTM com dado SARIMA prevê:",
    denormalize(predValue).toFixed(2)
  );
})();
