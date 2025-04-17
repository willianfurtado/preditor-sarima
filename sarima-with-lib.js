const fs = require("fs");
const parse = require("csv-parse/sync");
const ARIMA = require("arima");

//leitura do arquivo
const file = fs.readFileSync("./Cotacoes_Filtradas_nov_abril.csv");

const records = parse.parse(file, {
  columns: true,
  skip_empty_lines: true,
  delimiter: ",", 
});

//padronização dos dados pro SARIMA
const ts = records.map((r) => parseFloat(r["Último"])).reverse(); 

const arima = new ARIMA({
  p: 3,
  d: 1,
  q: 3,
  P: 1,
  D: 1,
  Q: 0,
  s: 5, 
  verbose: false,
}).train(ts); //R$327.64

const [pred] = arima.predict(1);

console.log("Previsão para amanhã:", pred[0].toFixed(2));

