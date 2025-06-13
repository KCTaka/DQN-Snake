class DynamicPlot {
    constructor(ctx) {
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Average Score',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    fill: false
                }, {
                    label: 'Score',
                    data: [],
                    borderColor: 'rgba(192, 75, 75, 1)',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                scales: {
                    x: { title: { display: true, text: 'Episodes' } },
                    y: { title: { display: true, text: 'Score' } }
                }
            }
        });
    }

    plot(avgScores, currentScores) {
        this.chart.data.labels = avgScores.map((_, i) => (i + 1) * 10); // Assuming plot every 10 episodes
        this.chart.data.datasets[0].data = avgScores;
        this.chart.data.datasets[1].data = currentScores;
        this.chart.update();
    }
}