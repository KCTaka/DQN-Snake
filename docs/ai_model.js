// ai_model.js
const session = new onnx.InferenceSession({ backendHint: 'webgl' });
const modelUrl = './snake_ai.onnx';
let isModelLoaded = false;

async function loadModel() {
    await session.loadModel(modelUrl);
    isModelLoaded = true;
    console.log("AI Model Loaded");
}

async function getAIMove(gameState) {
    if (!isModelLoaded) {
        return null;
    }
    // Assuming gameState is a flat array matching [1,C,H,W]
    const C = 4, H = 84, W = 84;
    const tensor = new onnx.Tensor(new Float32Array(gameState), 'float32', [1, C, H, W]);
    const outputMap = await session.run([tensor]);
    const outputTensor = outputMap.get('output');
    const data = outputTensor.data;
    let maxIdx = 0;
    for (let i = 1; i < data.length; i++) {
        if (data[i] > data[maxIdx]) maxIdx = i;
    }
    return maxIdx;
}

loadModel();
