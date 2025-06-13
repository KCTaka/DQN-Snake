// --- DOM Elements ---
const gameBoard = document.getElementById('game-board');
const ctx = gameBoard.getContext('2d');
const highScoreElement = document.getElementById('high-score');
const episodeCountElement = document.getElementById('episode-count');
const epsilonElement = document.getElementById('epsilon-value');
const avgScoreElement = document.getElementById('avg-score');
const speedSlider = document.getElementById('speed-slider');
const toggleHyperparametersBtn = document.getElementById('toggle-hyperparameters');
const hyperparametersPanel = document.getElementById('hyperparameters-panel');
const saveSessionBtn = document.getElementById('save-session');
const loadSessionInput = document.getElementById('load-session-input');
const layersInput = document.getElementById('layers-input');
const gammaInput = document.getElementById('gamma-input');
const lrInput = document.getElementById('lr-input');
const decayInput = document.getElementById('decay-input');
const restartButton = document.getElementById('restart-button');
const networkCanvas = document.getElementById('network-canvas');

// --- Game Constants & Variables ---
const gridSize = 8;
const cellSize = gameBoard.width / gridSize;
let gameSpeed = 100 - speedSlider.value;
let isTrainingActive = false; // Flag to control the main loop

// --- Network Visualizer Class ---
class NetworkVisualizer {
    constructor(canvas) {
        this.ctx = canvas.getContext('2d');
        this.canvas = canvas;
        this.shape = [];
    }

    updateShape(shape) {
        this.shape = shape;
    }

    draw(activations) {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.shape.length === 0 || activations.length === 0) return;

        const layerSpacing = (this.canvas.width * 0.8) / (this.shape.length - 1);
        const marginX = this.canvas.width * 0.1;
        const nodePositions = [];

        for (let i = 0; i < this.shape.length; i++) {
            const layerNodes = [];
            const nodeCount = this.shape[i];
            const isBlock = nodeCount > 20;
            const nodeSpacing = this.canvas.height / (isBlock ? 2 : nodeCount + 1);
            for (let j = 0; j < (isBlock ? 1 : nodeCount); j++) {
                layerNodes.push({ x: marginX + i * layerSpacing, y: isBlock ? this.canvas.height / 2 : (j + 1) * nodeSpacing });
            }
            nodePositions.push(layerNodes);
        }

        ctx.strokeStyle = 'rgba(100, 100, 100, 0.5)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i < nodePositions.length - 1; i++) {
            for (const startNode of nodePositions[i]) {
                for (const endNode of nodePositions[i + 1]) {
                    ctx.beginPath(); ctx.moveTo(startNode.x, startNode.y); ctx.lineTo(endNode.x, endNode.y); ctx.stroke();
                }
            }
        }

        for (let i = 0; i < this.shape.length; i++) {
            const activationLayer = activations[i] || [];
            const nodeCount = this.shape[i];

            let layerLabel = '';
            if (i === 0) layerLabel = `Input [${nodeCount}]`;
            else if (i === this.shape.length - 1) layerLabel = `Output [${nodeCount}]`;
            else layerLabel = `Hidden ${i} [${nodeCount}]`;

            ctx.fillStyle = 'white';
            ctx.textAlign = 'center';
            ctx.fillText(layerLabel, nodePositions[i][0].x, 15);

            if (nodeCount > 20) {
                const avgActivation = activationLayer.length > 0 ? activationLayer.reduce((a, b) => a + b, 0) / activationLayer.length : 0;
                ctx.fillStyle = `rgba(139, 195, 74, ${Math.abs(avgActivation) * 0.8 + 0.2})`;
                ctx.fillRect(nodePositions[i][0].x - 30, 25, 60, this.canvas.height - 50);
            } else {
                for (let j = 0; j < nodeCount; j++) {
                    ctx.beginPath();
                    ctx.arc(nodePositions[i][j].x, nodePositions[i][j].y, 8, 0, Math.PI * 2);
                    const activation = activationLayer[j] || 0;
                    const green = activation > 0 ? 175 : 80;
                    const red = activation < 0 ? 175 : 80;
                    ctx.fillStyle = `rgba(${red}, ${green}, 80, ${Math.abs(activation) * 0.8 + 0.2})`;
                    ctx.fill();
                }
            }
        }
    }
}


// --- RL Agent Class ---
class DQNAgent {
    constructor(stateSize, actionSize, hiddenLayers, hyperparameters) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.hiddenLayers = hiddenLayers;
        this.memory = new Array(1000000);
        this.memoryIndex = 0;
        this.memoryFilled = false;
        this.gamma = hyperparameters.gamma;
        this.learningRate = hyperparameters.learningRate;
        this.epsilonLinearDecay = hyperparameters.epsilonDecay;
        this.epsilon = 1.0;
        this.epsilonInitial = 1.0;
        this.epsilonMin = 0.00;
        this.model = this.createModel();
        this.activationModel = this.createActivationModel();
    }

    createModel() {
        const model = tf.sequential({ name: 'main' });
        model.add(tf.layers.dense({ units: this.hiddenLayers[0], inputShape: [this.stateSize], activation: 'relu', name: `hidden_0` }));
        for (let i = 1; i < this.hiddenLayers.length; i++) {
            model.add(tf.layers.dense({ units: this.hiddenLayers[i], activation: 'relu', name: `hidden_${i}` }));
        }
        model.add(tf.layers.dense({ units: this.actionSize, activation: 'linear', name: 'output' }));
        this.compileModel(model);
        return model;
    }

    createActivationModel() {
        const outputs = this.model.layers.slice(0, -1).map(layer => layer.output);
        return tf.model({ inputs: this.model.inputs, outputs: outputs });
    }

    compileModel(model) {
        model.compile({ optimizer: tf.train.adam(this.learningRate), loss: 'meanSquaredError' });
    }

    updateEpsilon(episode) {
        this.epsilon = Math.max(this.epsilonMin, this.epsilonInitial - this.epsilonLinearDecay * episode);
    }

    remember(stateData, action, reward, nextStateData, done) {
        this.memory[this.memoryIndex] = { state: stateData, action, reward, nextState: nextStateData, done };
        this.memoryIndex++;
        if (this.memoryIndex >= this.memory.length) {
            this.memoryIndex = 0;
            this.memoryFilled = true;
        }
    }

    act(stateTensor) {
        if (Math.random() <= this.epsilon) {
            const randomAction = Math.floor(Math.random() * this.actionSize);
            const activations = [Array.from(stateTensor.dataSync()), ...this.hiddenLayers.map(size => new Array(size).fill(0)), new Array(this.actionSize).fill(0)];
            return { action: randomAction, activations };
        }

        return tf.tidy(() => {
            const outputTensor = this.model.predict(stateTensor);
            let hiddenTensors = this.activationModel.predict(stateTensor);
            if (!Array.isArray(hiddenTensors)) hiddenTensors = [hiddenTensors];

            const action = outputTensor.argMax(1).dataSync()[0];
            const activations = [
                Array.from(stateTensor.dataSync()),
                ...hiddenTensors.map(t => Array.from(t.dataSync())),
                Array.from(outputTensor.dataSync())
            ];
            return { action, activations };
        });
    }

    // --- START CORRECTION ---
    dispose() {
        // The main model owns all the layers. Disposing of it is sufficient.
        // Disposing of the activationModel separately would cause a "double dispose" error.
        this.model.dispose();
    }
    // --- END CORRECTION ---

    async replay(batchSize) {
        const currentMemorySize = this.memoryFilled ? this.memory.length : this.memoryIndex;
        if (currentMemorySize < batchSize) return;

        const miniBatchIndices = new Set();
        while (miniBatchIndices.size < batchSize) {
            miniBatchIndices.add(Math.floor(Math.random() * currentMemorySize));
        }

        const miniBatch = Array.from(miniBatchIndices).map(i => this.memory[i]);
        const states = miniBatch.map(e => e.state);
        const nextStates = miniBatch.map(e => e.nextState ? e.nextState : new Float32Array(this.stateSize).fill(0));

        const statesTensor = tf.tensor2d(states, [batchSize, this.stateSize]);
        const nextStatesTensor = tf.tensor2d(nextStates, [batchSize, this.stateSize]);

        const { currentQValues, maxNextQ } = tf.tidy(() => {
            return {
                currentQValues: this.model.predict(statesTensor),
                maxNextQ: this.model.predict(nextStatesTensor).max(1)
            };
        });

        const targetQValuesData = await currentQValues.array();
        const maxNextQData = await maxNextQ.array();

        for (let i = 0; i < batchSize; i++) {
            const { action, reward, done } = miniBatch[i];
            if (done) {
                targetQValuesData[i][action] = reward;
            } else {
                targetQValuesData[i][action] = reward + this.gamma * maxNextQData[i];
            }
        }

        const targetTensor = tf.tensor2d(targetQValuesData, [batchSize, this.actionSize]);
        await this.model.fit(statesTensor, targetTensor, { epochs: 1, verbose: 0 });
        tf.dispose([statesTensor, nextStatesTensor, currentQValues, maxNextQ, targetTensor]);
    }
}

// --- Main Game & Agent Initialization ---
let snake, food, direction, score, gameOver, frameIter;
let episode = 0;
let scoreHistory = [];
let lastActivations = [];
let agent;
const networkVisualizer = new NetworkVisualizer(networkCanvas);

function initializeAgent(hiddenLayers) {
    if (!hiddenLayers) {
        hiddenLayers = layersInput.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
        if (hiddenLayers.length === 0) {
            alert("Invalid layer definition. Using default [128].");
            hiddenLayers.push(128);
            layersInput.value = "128";
        }
    }
    agent = new DQNAgent(11, 3, hiddenLayers, {
        gamma: parseFloat(gammaInput.value),
        learningRate: parseFloat(lrInput.value),
        epsilonDecay: parseFloat(decayInput.value)
    });
    networkVisualizer.updateShape([11, ...hiddenLayers, 3]);
}


// --- Core Game Logic ---
function resetGame() {
    snake = [{ x: Math.floor(gridSize / 2), y: Math.floor(gridSize / 2) }];
    direction = { x: 1, y: 0 };
    createFood();
    score = 0;
    gameOver = false;
    frameIter = 0;
}

function createFood() {
    while (true) {
        food = { x: Math.floor(Math.random() * gridSize), y: Math.floor(Math.random() * gridSize) };
        if (!snake.some(segment => segment.x === food.x && segment.y === food.y)) break;
    }
}

function getState() {
    const head = snake[0];
    const point_l = { x: head.x - 1, y: head.y }, point_r = { x: head.x + 1, y: head.y }, point_u = { x: head.x, y: head.y - 1 }, point_d = { x: head.x, y: head.y + 1 };
    const dir_l = direction.x === -1, dir_r = direction.x === 1, dir_u = direction.y === -1, dir_d = direction.y === 1;
    const isCollision = pt => pt.x < 0 || pt.x >= gridSize || pt.y < 0 || pt.y >= gridSize || snake.slice(1).some(s => s.x === pt.x && s.y === pt.y);
    return tf.tensor2d([[(dir_r && isCollision(point_r)) || (dir_l && isCollision(point_l)) || (dir_u && isCollision(point_u)) || (dir_d && isCollision(point_d)), (dir_u && isCollision(point_r)) || (dir_d && isCollision(point_l)) || (dir_l && isCollision(point_u)) || (dir_r && isCollision(point_d)), (dir_d && isCollision(point_r)) || (dir_u && isCollision(point_l)) || (dir_r && isCollision(point_u)) || (dir_l && isCollision(point_d)), dir_l, dir_r, dir_u, dir_d, food.x < head.x, food.x > head.x, food.y < head.y, food.y > head.y].map(Number)], [1, 11]);
}

function doAction(action) {
    const directions = [{ x: 0, y: -1 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: -1, y: 0 }]; // U, R, D, L
    const currentDirIndex = directions.findIndex(d => d.x === direction.x && d.y === direction.y);
    let newDirIndex;
    if (action === 0) newDirIndex = currentDirIndex;
    else if (action === 1) newDirIndex = (currentDirIndex + 1) % 4;
    else newDirIndex = (currentDirIndex + 3) % 4;
    direction = directions[newDirIndex];
}

async function gameStep() {
    if (!isTrainingActive || gameOver) return;
    frameIter++;
    const stateTensor = getState();
    const { action, activations } = agent.act(stateTensor);
    lastActivations = activations;
    doAction(action);
    const head = { x: snake[0].x + direction.x, y: snake[0].y + direction.y };
    snake.unshift(head);
    let reward = -0.1;
    if (head.x < 0 || head.x >= gridSize || head.y < 0 || head.y >= gridSize || snake.slice(1).some(s => s.x === head.x && s.y === head.y) || frameIter > 100 * (snake.length)) {
        gameOver = true;
        reward = -1;
    } else if (head.x === food.x && head.y === food.y) {
        score++;
        reward = 1;
        createFood();
    } else {
        snake.pop();
    }
    const stateData = stateTensor.dataSync();
    tf.dispose(stateTensor);
    let nextStateData = null;
    if (!gameOver) {
        const nextStateTensor = getState();
        nextStateData = nextStateTensor.dataSync();
        tf.dispose(nextStateTensor);
    }
    agent.remember(stateData, action, reward, nextStateData, gameOver);
    await agent.replay(256);
}

// --- Main Loop and Drawing ---
function draw() {
    ctx.clearRect(0, 0, gameBoard.width, gameBoard.height);
    for (let i = 0; i < snake.length; i++) {
        ctx.fillStyle = i === 0 ? '#4caf50' : '#8bc34a';
        ctx.fillRect(snake[i].x * cellSize, snake[i].y * cellSize, cellSize, cellSize);
    }
    ctx.fillStyle = '#f44336';
    ctx.fillRect(food.x * cellSize, food.y * cellSize, cellSize, cellSize);
    networkVisualizer.draw(lastActivations);
}

// --- Plotting Logic ---
const chartCtx = document.getElementById('score-chart').getContext('2d');
const scoreChart = new Chart(chartCtx, { type: 'line', data: { labels: [], datasets: [{ label: 'Average Score (100 eps)', data: [], borderColor: '#4caf50', tension: 0.1 }, { label: 'Score', data: [], borderColor: 'rgba(244, 67, 54, 0.5)', showLine: false, pointRadius: 3 }] }, options: { scales: { y: { beginAtZero: true } } } });

function updatePlot() {
    const last100scores = scoreHistory.slice(-100);
    const avgScore = last100scores.reduce((a, b) => a + b, 0) / last100scores.length || 0;
    avgScoreElement.textContent = avgScore.toFixed(2);
    if (episode % 10 === 0 && episode > 0) {
        scoreChart.data.labels.push(episode);
        scoreChart.data.datasets[0].data.push(avgScore);
        scoreChart.data.datasets[1].data.push(score);
        if (scoreChart.data.labels.length > 50) {
            scoreChart.data.labels.shift();
            scoreChart.data.datasets.forEach(dataset => dataset.data.shift());
        }
        scoreChart.update('none');
    }
}

function redrawChartFromHistory() {
    const labels = [];
    const avgData = [];
    const scoreData = [];
    for (let i = 0; i < scoreHistory.length; i++) {
        if (i % 10 === 0 && i > 0) {
            labels.push(i);
            const last100 = scoreHistory.slice(Math.max(0, i - 100), i);
            avgData.push(last100.reduce((a, b) => a + b, 0) / last100.length);
            scoreData.push(scoreHistory[i]);
        }
    }
    scoreChart.data.labels = labels.slice(-50);
    scoreChart.data.datasets[0].data = avgData.slice(-50);
    scoreChart.data.datasets[1].data = scoreData.slice(-50);
    scoreChart.update();
}

async function mainLoop() {
    if (!isTrainingActive) return; // Stop the loop if not active
    await gameStep();
    draw();
    if (gameOver) {
        scoreHistory.push(score);
        highScoreElement.textContent = Math.max(Number(highScoreElement.textContent), score);
        episode++;
        agent.updateEpsilon(episode);
        episodeCountElement.textContent = episode;
        epsilonElement.textContent = agent.epsilon.toFixed(3);
        updatePlot();
        resetGame();
    }
    if (isTrainingActive) { // Check again before queuing the next frame
        setTimeout(mainLoop, gameSpeed);
    }
}

// --- Event Listeners and Handlers ---
function startNewSession(isInitialStart = false) {
    isTrainingActive = false; // Stop any previous loop
    if (agent) agent.dispose();

    initializeAgent();

    episode = 0;
    scoreHistory = [];
    highScoreElement.textContent = '0';
    episodeCountElement.textContent = '0';
    epsilonElement.textContent = (1.0).toFixed(3);

    redrawChartFromHistory();
    resetGame();

    if (!isInitialStart) {
        alert('Training has been restarted with new settings.');
    }

    isTrainingActive = true;
    mainLoop();
}

speedSlider.addEventListener('input', e => gameSpeed = 101 - e.target.value);
toggleHyperparametersBtn.addEventListener('click', () => {
    const panel = hyperparametersPanel;
    panel.classList.toggle('hidden');
    toggleHyperparametersBtn.textContent = panel.classList.contains('hidden') ? 'Show Settings' : 'Hide Settings';
});

gammaInput.addEventListener('change', () => { if (agent) agent.gamma = parseFloat(gammaInput.value) });
decayInput.addEventListener('change', () => { if (agent) agent.epsilonLinearDecay = parseFloat(decayInput.value) });
lrInput.addEventListener('change', () => {
    if (agent) {
        agent.learningRate = parseFloat(lrInput.value);
        agent.compileModel(agent.model);
    }
});

restartButton.addEventListener('click', () => startNewSession(false));

saveSessionBtn.addEventListener('click', async () => {
    const modelArtifacts = await agent.model.save(tf.io.withSaveHandler(async (artifacts) => artifacts));
    const sessionData = {
        networkShape: agent.hiddenLayers,
        modelTopology: modelArtifacts.modelTopology,
        weightSpecs: modelArtifacts.weightSpecs,
        weightData: Array.from(new Uint8Array(modelArtifacts.weightData)),
        hyperparameters: { gamma: agent.gamma, learningRate: agent.learningRate, epsilonDecay: agent.epsilonLinearDecay },
        trainingState: { episode: episode, scoreHistory: scoreHistory }
    };
    const blob = new Blob([JSON.stringify(sessionData)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `snake-ai-session-ep${episode}.json`;
    a.click();
    URL.revokeObjectURL(url);
});

loadSessionInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (e) => {
        try {
            isTrainingActive = false; // Stop loop while loading
            const sessionData = JSON.parse(e.target.result);
            const { networkShape, modelTopology, weightSpecs, weightData, hyperparameters, trainingState } = sessionData;

            if (!networkShape || !modelTopology || !weightSpecs) {
                alert('Error: Loaded file is missing critical network architecture data.');
                startNewSession(true); // Restart to a clean state
                return;
            }

            if (agent) agent.dispose();

            layersInput.value = networkShape.join(', ');
            gammaInput.value = hyperparameters.gamma;
            lrInput.value = hyperparameters.learningRate;
            decayInput.value = hyperparameters.epsilonDecay;

            initializeAgent(networkShape);

            const weightBuffer = new Uint8Array(weightData).buffer;
            agent.model = await tf.loadLayersModel(tf.io.fromMemory(modelTopology, weightSpecs, weightBuffer));
            agent.compileModel(agent.model);
            agent.activationModel = agent.createActivationModel();

            episode = trainingState.episode;
            scoreHistory = trainingState.scoreHistory;
            agent.updateEpsilon(episode);
            episodeCountElement.textContent = episode;
            highScoreElement.textContent = Math.max(0, ...scoreHistory);
            redrawChartFromHistory();
            resetGame();
            alert('Session loaded successfully!');
            isTrainingActive = true; // Resume training
            mainLoop();
        } catch (error) {
            alert('Failed to load session file. It may be corrupted or in the wrong format.');
            console.error("Load error:", error);
            startNewSession(true); // Restart to a clean state
        }
    };
    reader.readAsText(file);
    event.target.value = '';
});

// --- Initial Setup & Start ---
startNewSession(true);