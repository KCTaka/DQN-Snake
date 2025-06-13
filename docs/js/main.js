// This file should be loaded as a module in index.html
// <script src="js/main.js" type="module"></script>

// Import what's needed if using modules, otherwise they are global
// For simplicity, this example assumes global scope from script tags

document.addEventListener('DOMContentLoaded', () => {
    const game = new SnakeGame('game-canvas', 20); // Larger grid
    const plotter = new DynamicPlot(document.getElementById('plot-canvas').getContext('2d'));

    const startBtn = document.getElementById('start-btn');
    const modeSelect = document.getElementById('mode-select');

    let trainer;
    let model;
    let getStateFunc;

    function setupTrainer() {
        const mode = modeSelect.value;
        if (mode === 'simple') {
            model = createFCNN(11, [256], 3);
            getStateFunc = getSimpleState;
        } else { // sensor
            model = createFCNN(26, [256, 128], 3);
            getStateFunc = getSensorState;
        }
        trainer = new DQNTrainer(model, game, getStateFunc);
    }

    let episode = 0;
    const scores = [];
    const avgScores = [];
    let epsilon = 1.0;

    async function trainLoop() {
        game.reset();
        let state = trainer.getState(game);

        while (!game.gameEnd) {
            const action = trainer.getAction(state, epsilon);
            game.doAction(action);
            const reward = game.step();
            const nextState = game.gameEnd ? null : trainer.getState(game);

            trainer.replayMemory.append([state, action, reward, nextState, game.gameEnd]);
            state = nextState;

            trainer.updateModel();

            game.render(); // Render game
            await new Promise(r => setTimeout(r, 20)); // Control speed
        }

        // After episode ends
        scores.push(game.score);
        epsilon = Math.max(0.01, epsilon * 0.995); // Epsilon decay

        document.getElementById('episode-count').textContent = episode + 1;
        document.getElementById('current-score').textContent = game.score;

        if (episode % 10 === 0 && episode > 0) {
            const last10Scores = scores.slice(-10);
            const avg = last10Scores.reduce((a, b) => a + b, 0) / last10Scores.length;
            avgScores.push(avg);
            document.getElementById('avg-score').textContent = avg.toFixed(2);
            plotter.plot(avgScores, scores);
        }

        episode++;
        requestAnimationFrame(trainLoop); // Next episode
    }

    startBtn.addEventListener('click', () => {
        setupTrainer();
        startBtn.disabled = true;
        modeSelect.disabled = true;
        requestAnimationFrame(trainLoop);
    });
});