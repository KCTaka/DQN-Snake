// A simplified version of get_game_info_from_sequence
function getGameInfo(game) {
    return {
        headPos: game.snake[0],
        snakePos: game.snake,
        foodPos: game.food,
        actionDir: game.action,
        gridSize: game.gridSize
    };
}

// Corresponds to binary_state
function getSimpleState(game) {
    const info = getGameInfo(game);
    const head = info.headPos;
    const point_l = { x: head.x - 1, y: head.y };
    const point_r = { x: head.x + 1, y: head.y };
    const point_u = { x: head.x, y: head.y - 1 };
    const point_d = { x: head.x, y: head.y + 1 };

    const dir_l = info.actionDir.x === -1;
    const dir_r = info.actionDir.x === 1;
    const dir_u = info.actionDir.y === -1;
    const dir_d = info.actionDir.y === 1;

    const state = [
        // Danger ahead
        (dir_r && game.isCollided(point_r)) || (dir_l && game.isCollided(point_l)) || (dir_u && game.isCollided(point_u)) || (dir_d && game.isCollided(point_d)),
        // Danger right
        (dir_u && game.isCollided(point_r)) || (dir_d && game.isCollided(point_l)) || (dir_l && game.isCollided(point_u)) || (dir_r && game.isCollided(point_d)),
        // Danger left
        (dir_d && game.isCollided(point_r)) || (dir_u && game.isCollided(point_l)) || (dir_r && game.isCollided(point_u)) || (dir_l && game.isCollided(point_d)),

        // Move direction
        dir_l, dir_r, dir_u, dir_d,

        // Food location 
        info.foodPos.x < head.x, // food left
        info.foodPos.x > head.x, // food right
        info.foodPos.y < head.y, // food up
        info.foodPos.y > head.y  // food down
    ];

    return state.map(Number); // Convert boolean to 0 or 1
}

// Corresponds to sensor_state (simplified)
function getSensorState(game) {
    // This is a more complex function to translate directly without a good vector math library.
    // For now, let's create a placeholder that returns a fixed-size array.
    // A full implementation would require ray-casting logic.
    const state = new Array(26).fill(0); // 13 sensors * 2 values (dist, type)
    // A true implementation would calculate distances to walls/food/body in different directions.
    return state;
}