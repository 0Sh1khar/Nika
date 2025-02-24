const BASE_URL = "http://127.0.0.1:5000"; // Flask backend

// Fetch Stock Data
async function fetchStockData() {
    let symbol = document.getElementById("symbol").value;
    let response = await fetch(`${BASE_URL}/stock?symbol=${symbol}`);
    let data = await response.json();

    document.getElementById("stock-info").innerHTML = `
        <h3>Price: $${data.data.current_price}</h3>
        <h4>Prediction: ${data.prediction}</h4>
    `;

    updateChart(data.data.history);
}

// Update Stock Chart
function updateChart(history) {
    let ctx = document.getElementById("stockChart").getContext("2d");
    new Chart(ctx, {
        type: "line",
        data: {
            labels: Array.from({length: history.length}, (_, i) => i + 1),
            datasets: [{
                label: "Stock Price",
                data: history,
                borderColor: "blue",
                fill: false
            }]
        }
    });
}

// Fetch Stock News
async function fetchStockNews() {
    let symbol = document.getElementById("news-symbol").value;
    let response = await fetch(`${BASE_URL}/news?symbol=${symbol}`);
    let data = await response.json();

    let newsHTML = data.news.map(news => `<p><strong>${news.title}</strong>: ${news.content}</p>`).join("");
    document.getElementById("news-section").innerHTML = newsHTML;

    document.getElementById("sentiment").innerText = `Sentiment: ${data.sentiment}`;
}
