let seed;
let isNewGame = true;

let m, b;
let points = [];

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

function setup() {
    init();
    createCanvas(windowWidth, windowHeight);

    // Initialise the gradient and y-intersection of the line randomly
    // (comparable to weights and  biases of a Neural Network)
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function draw() {
    background(dark);

    fill(light);
    for(let i = 0; i < points.length; i++) {
        const [x, y] = denormalisePosition(points[i][0], points[i][1]);
        ellipse(x, y, 15);
    }

    let xs = [];
    let ys = [];
    for(let i = 0; i < points.length; i++) {
        xs.push(points[i][0]);
        ys.push(points[i][1]);
    }

    if (xs.length > 1) {
        train(xs, ys);

        let ys_pred = predict(xs);

        let mse = loss(ys_pred, ys);
        mse.print();
    }
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
}
