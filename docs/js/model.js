class SimpleNet extends torch.nn.Module {
    constructor(inputSize, hiddenLayers, outputSize) {
        super();
        const layers = [];
        let inSize = inputSize;

        // Hidden layers
        for (const outSize of hiddenLayers) {
            layers.push(new torch.nn.Linear(inSize, outSize));
            layers.push(new torch.nn.ReLU());
            inSize = outSize;
        }

        // Output layer
        layers.push(new torch.nn.Linear(inSize, outputSize));
        // after building layers, register them:
        this.layers = layers;
    }

    forward(x) {
        for (const layer of this.layers) {
            x = layer.forward(x);
        }
        return x;
    }
}


function createFCNN(inputSize, hiddenLayers, outputSize) {
    return new SimpleNet(inputSize, hiddenLayers, outputSize);
}