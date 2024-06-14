import { useState } from "react";
import axios from "axios";
import "./App.css";
interface Input {
  Pregnancies: number | null;
  Glucose: number | null;
  BloodPressure: number | null;
  SkinThickness: number | null;
  Insulin: number | null;
  BMI: number | null;
  DiabetesPedigreeFunction: number | null;
  Age: number | null;
}
function App() {
  const options: string[] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
  ];
  const [inputs, setInputs] = useState<Input>({
    Pregnancies: 0.0,
    Glucose: 0.0,
    BloodPressure: 0.0,
    SkinThickness: 0.0,
    Insulin: 0.0,
    BMI: 0.0,
    DiabetesPedigreeFunction: 0.0,
    Age: 0.0,
  });
  const [result, setResult] = useState<string | undefined>(undefined);
  const handleChange = (name: keyof Input, value: number | null) => {
    setInputs((old) => ({ ...old, [name]: value }));
  };
  const handleSubmit = async () => {
    const data = [];
    for (let i in inputs) {
      const temp = inputs[i as keyof Input];
      data.push(temp);
    }
    await axios
      .post("http://localhost:8000/api/", { data })
      .then((res) => setResult(res.data.message))
      .catch((err)=>console.log(err))
    console.log(inputs);
  };
  return (
    <div className="container">
      <div className="content">
        <h1 className="title">Diabetes Predictor</h1>
        <div onSubmit={handleSubmit} className="form-container">
          {options.map((data: string, index: number) => (
            <div key={index} className="input-elements">
              <label htmlFor={data}>{data}</label>
              <input
                type="Number"
                name={data}
                id={data}
                value={inputs[(data as keyof Input) || ""]}
                onChange={(e) =>
                  handleChange(data as keyof Input, parseFloat(e.target.value))
                }
              />
            </div>
          ))}
        </div>
        <button onClick={handleSubmit}>Submit</button>
      </div>
      <div
        className="result-container"
        style={{
          opacity: result !== undefined ? 1 : 0,
          backgroundColor: result && result.includes("not") ? "green" : "red",
        }}
      >
        <h3 className="result">{result}</h3>
      </div>
    </div>
  );
}

export default App;
