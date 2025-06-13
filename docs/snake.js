const gameBoard = document.getElementById('game-board');
const ctx = gameBoard.getContext('2d');
const scoreElement = document.getElementById('score');
const highScoreElement = document.getElementById('high-score');
const episodeCountElement = document.getElementById('episode-count');
const epsilonElement = document.getElementById('epsilon-value');
const avgScoreElement = document.getElementById('avg-score');
const speedSlider = document.getElementById('speed-slider');

const gridSize = 8; // Matching python project
const cellSize = gameBoard.width / gridSize;
let gameSpeed = 100 - speedSlider.value; // Initial speed from slider

class DQNAgent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.memory = new Array(1000000);
        this.memoryIndex = 0;
        this.memoryFilled = false;

        this.gamma = 0.95;
        this.epsilon = 1.0;
        this.epsilonInitial = 1.0;
        this.epsilonMin = 0.00;
        this.epsilonLinearDecay = 1 / 5000;

        this.learningRate = 5e-5; // 0.00005
        this.model = this.createModel();
    }

    createModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 128, inputShape: [this.stateSize], activation: 'relu' }));
        model.add(tf.layers.dense({ units: this.actionSize, activation: 'linear' }));
        model.compile({ optimizer: tf.train.adam(this.learningRate), loss: 'meanSquaredError' });
        return model;
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
            return Math.floor(Math.random() * this.actionSize);
        }
        const prediction = this.model.predict(stateTensor);
        const action = prediction.argMax(1).dataSync()[0];
        tf.dispose(prediction);
        return action;
    }

    async replay(batchSize) {
        const currentMemorySize = this.memoryFilled ? this.memory.length : this.memoryIndex;
        if (currentMemorySize < batchSize) {
            return;
        }

        const miniBatchIndices = new Set();
        while (miniBatchIndices.size < batchSize) {
            miniBatchIndices.add(Math.floor(Math.random() * currentMemorySize));
        }

        const miniBatch = Array.from(miniBatchIndices).map(i => this.memory[i]);
        const states = miniBatch.map(e => e.state);
        const nextStates = miniBatch.map(e => e.nextState ? e.nextState : new Float32Array(this.stateSize).fill(0));

        const statesTensor = tf.tensor2d(states, [batchSize, this.stateSize]);
        const nextStatesTensor = tf.tensor2d(nextStates, [batchSize, this.stateSize]);

        // tf.tidy for synchronous predictions and return the needed tensors.
        const { currentQValues, maxNextQ } = tf.tidy(() => {
            const currentQ = this.model.predict(statesTensor);
            const nextQ = this.model.predict(nextStatesTensor);
            const maxNext = nextQ.max(1);
            return { currentQValues: currentQ, maxNextQ: maxNext };
        });

        // asynchronous data retrieval outside of tidy.
        const targetQValuesData = await currentQValues.array();
        const maxNextQData = await maxNextQ.array();

        // calculate the target Q-values
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

let snake, food, direction, relativeDirection, score, gameOver, frameIter;
let episode = 0;
const agent = new DQNAgent(11, 3);
let scoreHistory = [];
let avgScoreHistory = [];

function resetGame() {
    snake = [{ x: Math.floor(gridSize / 2), y: Math.floor(gridSize / 2) }];
    direction = { x: 1, y: 0 };
    relativeDirection = 1; // 0=left, 1=forward, 2=right
    createFood();
    score = 0;
    gameOver = false;
    frameIter = 0;
}

function createFood() {
    while (true) {
        food = {
            x: Math.floor(Math.random() * gridSize),
            y: Math.floor(Math.random() * gridSize)
        };
        if (!snake.some(segment => segment.x === food.x && segment.y === food.y)) {
            break;
        }
    }
}

function getState() {
    const head = snake[0];

    const point_l = { x: head.x - 1, y: head.y };
    const point_r = { x: head.x + 1, y: head.y };
    const point_u = { x: head.x, y: head.y - 1 };
    const point_d = { x: head.x, y: head.y + 1 };

    const dir_l = direction.x === -1;
    const dir_r = direction.x === 1;
    const dir_u = direction.y === -1;
    const dir_d = direction.y === 1;

    const isCollision = (pt) => pt.x < 0 || pt.x >= gridSize || pt.y < 0 || pt.y >= gridSize || snake.slice(1).some(s => s.x === pt.x && s.y === pt.y);

    const state = [
        // Danger Straight
        (dir_r && isCollision(point_r)) || (dir_l && isCollision(point_l)) || (dir_u && isCollision(point_u)) || (dir_d && isCollision(point_d)),
        // Danger Right
        (dir_u && isCollision(point_r)) || (dir_d && isCollision(point_l)) || (dir_l && isCollision(point_u)) || (dir_r && isCollision(point_d)),
        // Danger Left
        (dir_d && isCollision(point_r)) || (dir_u && isCollision(point_l)) || (dir_r && isCollision(point_u)) || (dir_l && isCollision(point_d)),
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        food.x < head.x, // Food left
        food.x > head.x, // Food right
        food.y < head.y, // Food up
        food.y > head.y  // Food down
    ];

    return tf.tensor2d([state.map(Number)], [1, 11]);
}

function doAction(action) {
    // 0=straight, 1=right turn, 2=left turn
    const directions = [{ x: 0, y: -1 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: -1, y: 0 }]; // U, R, D, L
    const currentDirIndex = directions.findIndex(d => d.x === direction.x && d.y === direction.y);

    let newDirIndex;
    if (action === 0) { // Straight
        newDirIndex = currentDirIndex;
    } else if (action === 1) { // Right turn
        newDirIndex = (currentDirIndex + 1) % 4;
    } else { // Left turn 
        newDirIndex = (currentDirIndex + 3) % 4;
    }
    direction = directions[newDirIndex];
}

async function gameStep() {
    if (gameOver) return;

    frameIter++;
    const stateTensor = getState();
    const action = agent.act(stateTensor);

    doAction(action);
    const head = { x: snake[0].x + direction.x, y: snake[0].y + direction.y };
    snake.unshift(head);

    let reward = -0.1; // default reward
    if (head.x < 0 || head.x >= gridSize || head.y < 0 || head.y >= gridSize || snake.slice(1).some(s => s.x === head.x && s.y === head.y) || frameIter > 100 * snake.length) {
        gameOver = true;
        reward = -1; // death reward
    } else if (head.x === food.x && head.y === food.y) {
        score++;
        reward = 1; // food reward
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


function draw() {
    ctx.clearRect(0, 0, gameBoard.width, gameBoard.height);
    for (let i = 0; i < snake.length; i++) {
        ctx.fillStyle = i === 0 ? '#4caf50' : '#8bc34a';
        ctx.fillRect(snake[i].x * cellSize, snake[i].y * cellSize, cellSize, cellSize);
    }
    ctx.fillStyle = '#f44336';
    ctx.fillRect(food.x * cellSize, food.y * cellSize, cellSize, cellSize);
}

const chartCtx = document.getElementById('score-chart').getContext('2d');
const scoreChart = new Chart(chartCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Average Score',
            data: [],
            borderColor: '#4caf50',
            tension: 0.1,
            yAxisID: 'y'
        }, {
            label: 'Score',
            data: [],
            borderColor: 'rgba(244, 67, 54, 0.5)',
            backgroundColor: 'rgba(244, 67, 54, 0.2)',
            showLine: false,
            pointRadius: 3,
            yAxisID: 'y'
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true,
                type: 'linear',
                position: 'left'
            }
        }
    }
});

function updatePlot() {
    const last100scores = scoreHistory.slice(-100);
    const avgScore = last100scores.reduce((a, b) => a + b, 0) / last100scores.length || 0;

    scoreChart.data.labels.push(episode);
    scoreChart.data.datasets[0].data.push(avgScore);
    scoreChart.data.datasets[1].data.push(score);

    if (scoreChart.data.labels.length > 50) {
        scoreChart.data.labels.shift();
        scoreChart.data.datasets.forEach(dataset => dataset.data.shift());
    }

    scoreChart.update();
    avgScoreElement.textContent = avgScore.toFixed(2);
}

async function mainLoop() {
    await gameStep();
    draw();

    if (gameOver) {
        scoreHistory.push(score);
        const currentHighScore = Number(highScoreElement.textContent);
        if (score > currentHighScore) {
            highScoreElement.textContent = score;
        }

        episode++;
        agent.updateEpsilon(episode);

        episodeCountElement.textContent = episode;
        epsilonElement.textContent = agent.epsilon.toFixed(3);
        if (episode % 10 === 0 && episode > 0) {
            updatePlot();
        }

        resetGame();
    }

    setTimeout(mainLoop, gameSpeed);
}

// --- Initial Setup & Event Listeners ---
speedSlider.addEventListener('input', (e) => {
    gameSpeed = 101 - e.target.value;
});

resetGame();
mainLoop();