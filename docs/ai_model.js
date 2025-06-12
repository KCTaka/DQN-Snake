// ai_model.js
let session;
const modelUrl = './snake_ai.onnx';
let isModelLoaded = false;

async function loadModel() {
    // Create session using onnxruntime-web
    session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    isModelLoaded = true;
    console.log("AI Model Loaded");
}

async function getAIMove(gameState) {
    if (!isModelLoaded) {
        return null;
    }
    // Assuming gameState is a flat array matching [1,C,H,W]
    const C = 4, H = 84, W = 84;
    const tensor = new ort.Tensor('float32', new Float32Array(gameState), [1, C, H, W]);
    const feeds = { input: tensor };
    const results = await session.run(feeds);
    const outputTensor = results.output;
    const data = outputTensor.data;
    let maxIdx = 0;
    for (let i = 1; i < data.length; i++) {
        if (data[i] > data[maxIdx]) maxIdx = i;
    }
    return maxIdx;
}

loadModel();
