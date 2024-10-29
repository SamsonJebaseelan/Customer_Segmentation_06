const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Parse JSON bodies (as sent by API clients)
app.use(express.json());

// Home route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Example API endpoint
app.post('/api/spending-score', (req, res) => {
    const { age, gender, purchaseFrequency, averageAmountSpent } = req.body;
    // Calculate spending score logic here
    const spendingScore = (purchaseFrequency * averageAmountSpent) / age; // Example logic
    res.json({ spendingScore });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
