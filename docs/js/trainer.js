class ReplayMemory {
    constructor(maxlen) {
        this.maxlen = maxlen;
        this.memory = [];
    }

    append(transition) {
        if (this.memory.length >= this.maxlen) {
            this.memory.shift();
        }
        this.memory.push(transition);
    }

    sample(batchSize) {
        const miniBatch = [];
        const indices = [];
        while (indices.length < batchSize) {
            const index = Math.floor(Math.random() * this.memory.length);
            if (!indices.includes(index)) {
                indices.push(index);
            }
        }
        for (const index of indices) {
            miniBatch.push(this.memory[index]);
        }
        return miniBatch;
    }

    get length() {
        return this.memory.length;
    }
}

class DQNTrainer {
    constructor(model, game, getStateFunc) {
        this.model = model;
        this.game = game;
        this.getState = getStateFunc;
        this.replayMemory = new ReplayMemory(10000);
        this.gamma = 0.9;
        this.optimizer = new torch.optim.Adam(this.model.parameters(), { lr: 0.001 });
        this.criterion = new torch.nn.MSELoss();
    }

    getAction(state, epsilon) {
        if (Math.random() < epsilon) {
            return Math.floor(Math.random() * 3); // 0, 1, or 2
        } else {
            const stateTensor = torch.tensor([state]);
            const qTensor = this.model.forward(stateTensor);
            // convert to JS array [[...]] and grab first row
            const qArr = qTensor.data[0];
            let maxIdx = 0;
            for (let i = 1; i < qArr.length; i++) {
                if (qArr[i] > qArr[maxIdx]) maxIdx = i;
            }
            return maxIdx;
        }
    }

    updateModel() {
        if (this.replayMemory.length < 100) return; // Wait for more samples

        const batchSize = 32;
        const miniBatch = this.replayMemory.sample(batchSize);

        const states = miniBatch.map(t => t[0]);
        const actions = miniBatch.map(t => t[1]);
        const rewards = miniBatch.map(t => t[2]);
        const nextStates = miniBatch.map(t => t[3]);
        const terminals = miniBatch.map(t => t[4]);

        const stateTensor = torch.tensor(states);
        const qValues = this.model.forward(stateTensor);

        const y = [];
        for (let i = 0; i < batchSize; i++) {
            if (terminals[i]) {
                y.push(rewards[i]);
            } else {
                // compute next-Q for this single sample via toArray()
                const nextQT = this.model.forward(torch.tensor([nextStates[i]]));
                const nextQArr = nextQT.data[0];
                let rowMax = nextQArr[0];
                for (let j = 1; j < nextQArr.length; j++) {
                    if (nextQArr[j] > rowMax) rowMax = nextQArr[j];
                }
                y.push(rewards[i] + this.gamma * rowMax);
            }
        }

        // materialize qValues as JS array, inject targets, then re-create tensor
        const qArr2D = qValues.data;
        for (let i = 0; i < batchSize; i++) {
            qArr2D[i][actions[i]] = y[i];
        }
        const targetQValues = torch.tensor(qArr2D);

        const loss = this.criterion.forward(qValues, targetQValues);
        this.optimizer.zero_grad();
        loss.backward();
        this.optimizer.step();
    }
}