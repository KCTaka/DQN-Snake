class SnakeGame {
    constructor(canvasId, gridSize = 10) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.gridSize = gridSize;
        this.cellSize = this.canvas.width / this.gridSize;
        this.deathReward = -1;
        this.foodReward = 1;
        this.reset();
    }

    reset() {
        this.snake = [{ x: 5, y: 5 }];
        this.action = { x: 0, y: 0 };
        this.food = this._generateFood();
        this.score = 0;
        this.gameEnd = false;
        this.frameIter = 0;
    }

    _generateFood() {
        let foodPos;
        do {
            foodPos = {
                x: Math.floor(Math.random() * this.gridSize),
                y: Math.floor(Math.random() * this.gridSize)
            };
        } while (this.snake.some(segment => segment.x === foodPos.x && segment.y === foodPos.y));
        return foodPos;
    }

    isCollided(pos) {
        if (pos.x < 0 || pos.x >= this.gridSize || pos.y < 0 || pos.y >= this.gridSize) {
            return true;
        }
        return this.snake.slice(1).some(segment => segment.x === pos.x && segment.y === pos.y);
    }

    // Action: 0=Left, 1=Forward, 2=Right (relative to snake direction)
    doAction(action) {
        const actions = {
            // [y, -x] for left turn, [-y, x] for right turn
            0: { x: this.action.y, y: -this.action.x }, // Left
            1: { x: this.action.x, y: this.action.y }, // Forward
            2: { x: -this.action.y, y: this.action.x }  // Right
        };

        if (this.action.x === 0 && this.action.y === 0) { // Initial move
            this.action = { x: 0, y: -1 }; // Move up initially
        } else if (action !== 1) { // Not forward
            this.action = actions[action];
        }
    }


    step() {
        this.frameIter++;
        let reward = 0;
        const newHead = { x: this.snake[0].x + this.action.x, y: this.snake[0].y + this.action.y };
        this.snake.unshift(newHead);

        if (this.isCollided(newHead) || this.frameIter > 100 * (this.snake.length + 1)) {
            reward = this.deathReward;
            this.gameEnd = true;
            return reward;
        }

        if (newHead.x === this.food.x && newHead.y === this.food.y) {
            reward = this.foodReward;
            this.score++;
            this.food = this._generateFood();
        } else {
            this.snake.pop();
        }

        return reward;
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw food
        this.ctx.fillStyle = 'red';
        this.ctx.fillRect(this.food.x * this.cellSize, this.food.y * this.cellSize, this.cellSize, this.cellSize);

        // Draw snake
        this.ctx.fillStyle = 'green';
        this.snake.forEach(segment => {
            this.ctx.fillRect(segment.x * this.cellSize, segment.y * this.cellSize, this.cellSize, this.cellSize);
        });
    }

    // Get a grid representation of the game state
    getImage() {
        const image = [new Array(this.gridSize).fill(0).map(() => new Array(this.gridSize).fill(0)),
        new Array(this.gridSize).fill(0).map(() => new Array(this.gridSize).fill(0))];
        this.snake.forEach(seg => image[0][seg.y][seg.x] = 1);
        image[1][this.food.y][this.food.x] = 1;
        return image;
    }
}