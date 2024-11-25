const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const port = 5000;

app.use(cors());
app.use(bodyParser.json());

// API endpoint to handle prediction requests
app.post('/predict', (req, res) => {
  const { year, district, population, week, rainsum, meantemperature } = req.body;

  // Pass input data to the model script
  const pythonProcess = spawn('python', ['predict_model.py', year, district, population, week, rainsum, meantemperature]);

  pythonProcess.stdout.on('data', (data) => {
    res.json({ predicted_cases: data.toString() });
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      res.status(500).send('Error in prediction');
    }
  });
});

// Serve static files (if you decide to host the frontend here)
app.use(express.static(path.join(__dirname, 'frontend/build')));

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
