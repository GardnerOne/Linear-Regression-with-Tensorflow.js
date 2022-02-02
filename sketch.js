let seed;
let isNewGame = true;

let m, b;
let points;

let learningRate;
let optimizer;
let mse;

function setup() {
    init();
    createCanvas(windowWidth, windowHeight);

    points = [];

    learningRate = 0.1;
    mse = Infinity;
    optimizer = tf.train.sgd(learningRate);

    // Initialise the gradient and y-intersection of the line randomly
    // (comparable to weights and  biases of a Neural Network)
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function draw() {
    background(dark);

    fill(light);
    for (let i = 0; i < points.length; i++) {
        const [x, y] = denormalisePosition(points[i][0], points[i][1]);
        ellipse(x, y, 15);
    }

    let xs = [];
    let ys = [];
    for (let i = 0; i < points.length; i++) {
        xs.push(points[i][0]);
        ys.push(points[i][1]);
    }

    showBestFit();

    // Make sure we have sufficient data points before training
    if (xs.length > 1) {
        // Auto-dispose generated tensors
        tf.tidy(() => {
            train(xs, ys);
            let ys_pred = predict(xs);
            mse = loss(ys_pred, ys).dataSync();
        });

        learningRate -= learningRate / 10000;
    }

    // Debug info
    text(`Learning rate: ${learningRate}`, 20, 20);
    text(`Loss: ${mse}`, 20, 40);
    text(`Tensors: ${tf.memory().numTensors}`, 20, 60);
}

function predict(xs) {

    const tfxs = tf.tensor1d(xs);
    const tfys_pred = tfxs.mul(m).add(b);

    return tfys_pred;
}

function loss(predictions, labels) {
    // Calculate the mean squared error
    return predictions.sub(labels).square().mean();
}

function train(xs, ys) {
    const tfys = tf.tensor1d(ys);
    optimizer.minimize(() => loss(predict(xs), tfys));
}

function showBestFit() {
    // Show the line of best fit across the graph, 0 to width (or 0 to 1 normalised)
    const x_bounds = [0, 1];
    const tf_y_bounds_pred = tf.tidy(() => predict(x_bounds));
    const y_bounds_pred = tf_y_bounds_pred.dataSync();

    const [x0, y0] = denormalisePosition(x_bounds[0], y_bounds_pred[0]);
    const [x1, y1] = denormalisePosition(x_bounds[1], y_bounds_pred[1]);

    tf_y_bounds_pred.dispose();

    stroke(primary);
    strokeWeight(2);
    line(x0, y0, x1, y1);
    noStroke();
}

function normalisePosition(x, y) {
    // Normalise the data, making sure the y-axis direction is up
    const a = map(x, 0, width, 0, 1);
    const b = map(y, 0, height, 1, 0);

    return [a, b];
}

function denormalisePosition(x, y) {
    const a = map(x, 0, 1, 0, width);
    const b = map(y, 1, 0, 0, height);

    return [a, b];
}

function mousePressed() {
    points.push(normalisePosition(mouseX, mouseY))
}

function keyPressed() {
    // Space Key
    switch (keyCode) {
        case 32: // SPACE
            togglePlay();
            break;
        case 78: // N
            nextFrame();
            break;
        case 81: // Q
            reset(true);
            break;
        case 82: // R
            reset();
            break;
        default:
            break;
    }
}

function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
}

// Helpers
/**
 * Initialise the animation.
 *
 * For new games, print the animation controls.
 * */
function init() {
    if (isNewGame) {
        console.clear();
        seed = random(Number.MAX_SAFE_INTEGER);

        print('Controls:');
        print('"SPACE" - Toggle Play/Pause');
        print('    "N" - Next frame (when paused)');
        print('    "R" - Reset world');
        print('    "Q" - New game');
    }
    randomSeed(seed)
}

/** Pause or resume the animation */
function togglePlay() {
    if (isLooping()) {
        noLoop();
        print('Paused');
    } else {
        loop();
        print('Resume')
    }
}


/** Move forward one frame at a time */
function nextFrame() {
    if (isLooping()) return;

    redraw();
    print('Next frame')
}

/**
 * Reset the game to its initial state.
 *
 * By hard resetting, the seed for the game will be refreshed, creating a new world.
 * @param {Boolean} hard Whether a new session should be created or not (default = false)
 */
function reset(hard = false) {
    isNewGame = hard;
    setup();
    redraw();
    hard ? print('New game') : print('Reset world');

    // Clear any tensors from memory
    tf.dispose();
    // Clear any tensor variables from memory
    tf.disposeVariables();
}
