import React, { useState } from 'react';
import axios from 'axios';
import './App.css';  // Import the CSS file

function App() {
  const [year, setYear] = useState(2024);
  const [district, setDistrict] = useState(1);
  const [population, setPopulation] = useState(100000);
  const [week, setWeek] = useState(1);
  const [rainsum, setRainsum] = useState(0);
  const [meantemperature, setMeantemperature] = useState(20);
  const [predictedCases, setPredictedCases] = useState(null);

  const handlePrediction = async () => {
    try {
      const response = await axios.post('http://localhost:5000/predict', {
        year,
        district,
        population,
        week,
        rainsum,
        meantemperature
      });
      setPredictedCases(response.data.predicted_cases);
    } catch (error) {
      console.error('Error in prediction', error);
    }
  };

  return (
    <div className="App">
      <h1>Weekly Hospitalized Cases Forcast</h1>
      
      <div>
        <label>Year:</label>
        <input type="number" value={year} onChange={(e) => setYear(e.target.value)} />
      </div>
      <div>
        <label>District:</label>
        <input type="number" value={district} onChange={(e) => setDistrict(e.target.value)} />
      </div>
      <div>
        <label>Population:</label>
        <input type="number" value={population} onChange={(e) => setPopulation(e.target.value)} />
      </div>
      <div>
        <label>Week:</label>
        <input type="number" value={week} onChange={(e) => setWeek(e.target.value)} />
      </div>
      <div>
        <label>Total Rainfall (mm):</label>
        <input type="number" value={rainsum} onChange={(e) => setRainsum(e.target.value)} />
      </div>
      <div>
        <label>Mean Temperature (°C):</label>
        <input type="number" value={meantemperature} onChange={(e) => setMeantemperature(e.target.value)} />
      </div>
      
      <button onClick={handlePrediction}>Predict</button>

      {predictedCases !== null && (
        <div>
          <h2>No of Weekly Hospitalized Cases: {predictedCases}</h2>
        </div>
      )}
    </div>
  );
}

export default App;
